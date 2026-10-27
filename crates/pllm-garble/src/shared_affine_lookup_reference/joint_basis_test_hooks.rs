//! Test-only downstream injection and destructor-ENTRY observations. No owner,
//! lease, refund, callback, lock or allocator lives in this controller. Numeric
//! observations do not establish completed erasure or physical deallocation.
use crate::{shared_allocation::Budget, shared_rescale_reference::Error};
use std::{cell::{Cell, RefCell}, marker::PhantomData, rc::Rc, sync::Weak};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    JointEntropy,
    BoundaryReserve,
    BoundaryCapacity,
    CoefficientReserve,
    CoefficientCapacity,
    DcfReserve,
    DcfCapacity,
    DcfEntropy,
    RingEntropy,
    TripleEntropy,
    KeyDrop,
    CorrectionDrop,
    TripleDrop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Event {
    pub kind: Kind,
    // Public lengths/capacities only, never secret material or borrowed views.
    pub amount: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fault {
    pub kind: Kind,
    pub ordinal: usize,
    pub prefix: usize,
    pub unwind: bool,
}
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct InjectedPanic(pub Fault);
pub(crate) fn assert_injected<T>(result: &std::thread::Result<T>, fault: Fault) {
    if result.as_ref().err().and_then(|payload| payload.downcast_ref::<InjectedPanic>())
        != Some(&InjectedPanic(fault)) {
        std::panic::panic_any(EvidenceMismatch::PanicIdentity);
    }
}
fn inject(fault: Fault) {
    if fault.unwind {
        std::panic::panic_any(InjectedPanic(fault));
    }
}

pub(crate) struct Config {
    pub counter: Weak<Budget>,
    pub expected_identity: usize,
    // Exact full live charge for THIS phase, including bootstrap/source. A new
    // guard is installed after explicitly dropping exported frames/outputs.
    pub expected_charge: usize,
    pub expected_updates: usize,
    pub fault: Option<Fault>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Record {
    pub event: Event,
    pub identity: usize,
    pub used: usize,
    pub expected: usize,
    pub updates: usize,
    pub expected_updates: usize,
}

pub(crate) const MAX_EVENTS: usize = 96;
#[derive(Clone, Copy)]
pub(crate) struct Observation {
    pub records: [Option<Record>; MAX_EVENTS],
    pub len: usize,
    pub failed: bool,
    // Bytes copied by the deterministic reader, NOT requested draw lengths.
    pub consumed: [usize; 5],
}
impl Observation {
    fn empty() -> Self {
        Self { records: [None; MAX_EVENTS], len: 0, failed: false, consumed: [0; 5] }
    }
}
struct State {
    config: Config,
    observation: Observation,
}
thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
    // Also writable if the RefCell is already borrowed. Observation never panics
    // on reentrancy/overflow, particularly while an actual owner is unwinding.
    static FAILED: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn active() -> bool {
    STATE.with(|state| state.try_borrow().map_or(true, |state| state.is_some()))
}

pub(crate) struct Guard {
    previous: Option<State>,
    previous_failed: bool,
    // TLS guard may neither move to another thread nor own any counter handle.
    _same_thread: PhantomData<Rc<()>>,
}
pub(crate) fn install(config: Config) -> Guard {
    // Check ALL controllers before modifying ANY controller. These assertions
    // concern fixture installation only, never observer/destructor execution.
    assert!(!crate::shared_wide_product::ring::test_source_active());
    assert!(!super::joint_basis::test_source_active());
    assert!(!active());
    STATE.with(|state| {
        let mut state = state.try_borrow_mut().expect("hook installation reentered");
        let previous = state.replace(State { config, observation: Observation::empty() });
        Guard { previous, previous_failed: FAILED.with(|failed| failed.replace(false)),
            _same_thread: PhantomData }
    })
}
impl Guard {
    pub(crate) fn issuance_complete(&self) {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            let state = state.as_mut().unwrap();
            // Successful issue drops its zero-byte residual Lease: one update.
            // No reserve/release is permitted during the private owner moves.
            let expected = state.config.expected_updates.checked_add(1).unwrap();
            assert_eq!(state.config.counter.upgrade().unwrap().test_accounting_updates(), expected);
            state.config.expected_updates = expected;
        });
    }

    pub(crate) fn snapshot(&self) -> Observation {
        let mut observation = STATE.with(|state| {
            state.try_borrow().ok().and_then(|state| {
                state.as_ref().map(|state| state.observation)
            }).unwrap_or_else(|| {
                let mut observation = Observation::empty();
                observation.failed = true;
                observation
            })
        });
        observation.failed |= FAILED.with(Cell::get);
        observation
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        STATE.with(|state| {
            if let Ok(mut state) = state.try_borrow_mut() {
                *state = self.previous.take();
                FAILED.with(|failed| failed.set(self.previous_failed));
            } else {
                FAILED.with(|failed| failed.set(true));
            }
        });
    }
}

// None means inactive; Some(None) means active without injection at this event.
// Any temporary upgrade is dropped HERE, before returning or injecting panic.
fn record(event: Event) -> Option<Option<Fault>> {
    STATE.with(|state| {
        let Ok(mut state) = state.try_borrow_mut() else {
            FAILED.with(|failed| failed.set(true));
            return None;
        };
        let state = state.as_mut()?;
        let observation = &mut state.observation;
        if observation.len == MAX_EVENTS {
            observation.failed = true;
            return Some(None);
        }
        let ordinal = observation.records[..observation.len].iter().flatten()
            .filter(|record| record.event.kind == event.kind).count();
        let snapshot = state.config.counter.upgrade().map(|counter| {
            let (identity, used) = counter.test_snapshot();
            (identity, used, counter.test_accounting_updates())
        });
        let (identity, used, updates) = snapshot.unwrap_or((0, 0, 0));
        observation.failed |= snapshot.is_none()
            || identity != state.config.expected_identity || used != state.config.expected_charge
            || updates != state.config.expected_updates;
        observation.records[observation.len] = Some(Record {
            event, identity, used, expected: state.config.expected_charge,
            updates, expected_updates: state.config.expected_updates,
        });
        observation.len += 1;
        Some(state.config.fault.filter(|fault| fault.kind == event.kind && fault.ordinal == ordinal))
    })
}
pub(crate) fn observe(event: Event) {
    let _ = record(event);
}
pub(crate) fn reserve(kind: Kind, amount: usize) -> Result<(), Error> {
    if let Some(Some(fault)) = record(Event { kind, amount }) {
        // record()'s RefMut AND temporary counter Arc are already gone.
        inject(fault);
        return Err(Error::Resource);
    }
    Ok(())
}
// A real bounded Read implementation: read_exact observes a short source and
// fails only after consuming its available prefix. No requested-size ledger.
struct PrefixReader { remaining: usize, consumed: usize, byte: u8 }
impl std::io::Read for PrefixReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let count = out.len().min(self.remaining);
        out[..count].fill(self.byte);
        self.remaining -= count;
        self.consumed += count;
        Ok(count)
    }
}
fn entropy_index(kind: Kind, observation: &Observation) -> usize {
    match kind {
        Kind::JointEntropy => 0,
        Kind::DcfEntropy => observation.records[..observation.len].iter().flatten()
            .filter(|record| record.event.kind == Kind::DcfEntropy).count(),
        Kind::RingEntropy => 3, Kind::TripleEntropy => 4,
        _ => panic!("not entropy"),
    }
}
fn consume(kind: Kind, bytes: &mut [u8], available: usize, byte: u8) {
    let mut reader = PrefixReader { remaining: available, consumed: 0, byte };
    let result = std::io::Read::read_exact(&mut reader, bytes);
    assert_eq!(result.is_ok(), available >= bytes.len());
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let observation = &mut state.as_mut().unwrap().observation;
        let index = entropy_index(kind, observation);
        observation.consumed[index] += reader.consumed;
    });
}
pub(crate) fn fill(kind: Kind, bytes: &mut [u8]) -> Option<Result<(), Error>> {
    let fault = record(Event { kind, amount: bytes.len() })?;
    if bytes.iter().any(|&byte| byte != 0) {
        FAILED.with(|failed| failed.set(true));
    }
    if let Some(fault) = fault {
        if fault.prefix > bytes.len() {
            FAILED.with(|failed| failed.set(true));
            return Some(Err(Error::Randomness));
        }
        consume(kind, bytes, fault.prefix, 0xa5);
        // No TLS borrow spans the injected panic or subsequent owner destruction.
        inject(fault);
        Some(Err(Error::Randomness))
    } else {
        consume(kind, bytes, bytes.len(), 0x39);
        Some(Ok(()))
    }
}

#[test]
fn bounded_observer_rejects_overflow_reentry_missing_counter_and_wrong_snapshots() {
    use crate::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};
    let owned = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (identity, used) = owned.budget().test_snapshot();
    let config = || Config { counter: owned.test_counter_weak(), expected_identity: identity,
        expected_charge: used, expected_updates: owned.budget().test_accounting_updates(),
        fault: None };
    let event = Event { kind: Kind::KeyDrop, amount: 0 };
    {
        let guard = install(config());
        for _ in 0..=MAX_EVENTS { observe(event); }
        let snapshot = guard.snapshot();
        assert_eq!(snapshot.len, MAX_EVENTS);
        assert!(snapshot.failed);
        assert_rejected(EvidenceMismatch::Observation, || assert_valid(&snapshot));
    }
    {
        let guard = install(config());
        STATE.with(|state| { let _borrow = state.borrow_mut(); observe(event); });
        assert!(guard.snapshot().failed);
        assert_rejected(EvidenceMismatch::Observation, || assert_valid(&guard.snapshot()));
        assert_eq!(guard.snapshot().len, 0);
    }
    for invalid in 0..4 {
        let mut config = config();
        match invalid {
            0 => config.counter = Weak::new(),
            1 => config.expected_identity ^= 1,
            2 => config.expected_charge += 1,
            _ => config.expected_updates += 1,
        }
        let guard = install(config);
        observe(event);
        assert_eq!(guard.snapshot().len, 1);
        assert!(guard.snapshot().failed);
        assert_rejected(EvidenceMismatch::Observation, || assert_valid(&guard.snapshot()));
    }
    let guard = install(config());
    observe(event);
    assert!(!guard.snapshot().failed); // prior failed controller was restored away
    assert_eq!(guard.snapshot().len, 1);
}

#[test]
fn scoped_guard_unwind_restores_inactive_flags_and_fresh_controller() {
    use crate::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};
    let owned = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let (identity, used) = owned.budget().test_snapshot();
    let fault = Fault { kind: Kind::JointEntropy, ordinal: 0, prefix: 1, unwind: true };
    let config = |fault| Config { counter: owned.test_counter_weak(),
        expected_identity: identity, expected_charge: used,
        expected_updates: owned.budget().test_accounting_updates(), fault };
    assert!(!active());
    assert!(!FAILED.with(Cell::get));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = install(config(Some(fault)));
        // Preexisting failure flag must be restored, not leaked by unwind.
        FAILED.with(|failed| failed.set(true));
        let _ = fill(Kind::JointEntropy, &mut [0; 48]);
    }));
    assert_injected(&result, fault);
    assert!(!active());
    assert!(!FAILED.with(Cell::get));
    let guard = install(config(None));
    assert!(fill(Kind::JointEntropy, &mut [0; 48]).unwrap().is_ok());
    assert!(!guard.snapshot().failed);
    assert_eq!(guard.snapshot().len, 1);
    assert_eq!(guard.snapshot().consumed, [48, 0, 0, 0, 0]);
    drop(guard);
    // Nested configured controllers are rejected, not restorable installs.
    // No TLS teardown or outstanding-RefCell-borrow restoration claim.
    FAILED.with(|failed| failed.set(true));
    { let guard = install(config(None)); assert!(!guard.snapshot().failed); }
    assert!(FAILED.with(Cell::get));
    FAILED.with(|failed| failed.set(false));
}

#[test]
fn unchanged_usage_does_not_hide_original_counter_release_and_reacquire() {
    use crate::shared_allocation::{OwnedBudget, MAX_ALLOCATION_BYTES};
    let owned = OwnedBudget::new(MAX_ALLOCATION_BYTES).unwrap();
    let lease = owned.budget().reserve(17).unwrap();
    let (identity, used) = owned.budget().test_snapshot();
    let updates = owned.budget().test_accounting_updates();
    let guard = install(Config { counter: owned.test_counter_weak(),
        expected_identity: identity, expected_charge: used, expected_updates: updates,
        fault: None });
    drop(lease);
    let replacement = owned.budget().reserve(17).unwrap();
    observe(Event { kind: Kind::KeyDrop, amount: 0 });
    let snapshot = guard.snapshot();
    let record = snapshot.records[0].unwrap();
    assert_eq!(record.identity, identity);
    assert_eq!(record.used, used);
    assert_eq!(record.updates, updates + 2);
    assert_ne!(record.updates, record.expected_updates);
    assert!(snapshot.failed); // then exercise the same positive assertion below
    assert_rejected(EvidenceMismatch::Observation, || assert_valid(&snapshot));
    drop(guard);
    drop(replacement);
}

#[test]
fn panic_identity_and_consumption_discriminators_reject_wrong_evidence() {
    let fault = Fault { kind: Kind::JointEntropy, ordinal: 0, prefix: 1, unwind: true };
    let unrelated: std::thread::Result<()> = Err(Box::new("unrelated"));
    assert_rejected(EvidenceMismatch::PanicIdentity, || assert_injected(&unrelated, fault));
    let wrong: std::thread::Result<()> = Err(Box::new(InjectedPanic(Fault { prefix: 2, ..fault })));
    assert_rejected(EvidenceMismatch::PanicIdentity, || assert_injected(&wrong, fault));
    let mut reader = PrefixReader { remaining: 1, consumed: 0, byte: 0xa5 };
    let mut bytes = [0; 48];
    assert!(std::io::Read::read_exact(&mut reader, &mut bytes).is_err());
    assert_eq!(reader.consumed, 1);
    assert_ne!(reader.consumed, bytes.len());
    assert_rejected(EvidenceMismatch::Consumption, ||
        assert_consumption([reader.consumed, 0, 0, 0, 0], [48, 0, 0, 0, 0]));
    assert_eq!(bytes[0], 0xa5);
    assert!(bytes[1..].iter().all(|&byte| byte == 0));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EvidenceMismatch { Observation, Consumption, PanicIdentity }
pub(crate) fn assert_valid(snapshot: &Observation) {
    if snapshot.failed { std::panic::panic_any(EvidenceMismatch::Observation); }
}
pub(crate) fn assert_consumption(actual: [usize; 5], expected: [usize; 5]) {
    if actual != expected { std::panic::panic_any(EvidenceMismatch::Consumption); }
}
fn assert_rejected(expected: EvidenceMismatch, check: impl FnOnce()) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(check));
    let payload = result.expect_err("negative control accepted wrong evidence");
    assert_eq!(payload.downcast_ref::<EvidenceMismatch>(), Some(&expected),
        "unrelated panic in negative control");
}
pub(crate) fn assert_install_conflict<T>(result: std::thread::Result<T>, expected: &str) {
    let payload = result.err().expect("controller conflict unexpectedly accepted");
    let message = payload.downcast_ref::<&'static str>().copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str));
    assert_eq!(message, Some(expected), "unrelated panic in controller conflict");
}
