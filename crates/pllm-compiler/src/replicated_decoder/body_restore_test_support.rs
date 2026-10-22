//! Test-only fault/observation hooks for the two REAL restoration callers.
//! Mounted separately in each adapter: no runtime, drain or proof issuer here.
use crate::replicated_decoder::{Error, MAX_PROGRAM_BYTES};
use std::cell::RefCell;
use std::sync::{Arc, Weak};

#[derive(Clone)]
pub(in crate::replicated_decoder) struct Owners {
    compiled: Weak<super::Compiled>,
    plan: Weak<super::Plan>,
    program: Weak<crate::replicated_decoder::Program>,
}

pub(in crate::replicated_decoder) fn owners(invocation: &super::Invocation) -> Owners {
    Owners {
        compiled: Arc::downgrade(&invocation.compiled),
        plan: Arc::downgrade(&invocation.compiled.plan),
        program: Arc::downgrade(&invocation.compiled.program),
    }
}

impl Owners {
    pub(in crate::replicated_decoder) fn counts_after_erasure(&self) -> [usize; 3] {
        // Tests relinquish ALL external strong owners before restoration. After
        // Values drops, only Compiled holds Plan; Worker and Compiled still hold
        // Program. This also rejects a Compiled Arc clone during restoration.
        // Record, don't assert inside Drop: a regression during an existing
        // reader/replay unwind must fail the test, not double-panic/abort it.
        [
            self.compiled.strong_count(),
            self.plan.strong_count(),
            self.program.strong_count(),
        ]
    }

    pub(in crate::replicated_decoder) fn assert_released(&self) {
        assert_eq!(self.compiled.strong_count(), 0);
        assert_eq!(self.plan.strong_count(), 0);
        assert_eq!(self.program.strong_count(), 0);
    }
}

#[derive(Clone, Copy)]
pub(in crate::replicated_decoder) enum Fault {
    None,
    ReadError,
    ReadPanic,
}

struct State {
    limit: usize,
    fault: Fault,
    extractions: usize,
    reads: usize,
    erased: Option<Box<dyn FnOnce()>>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

pub(in crate::replicated_decoder) struct Session;

pub(in crate::replicated_decoder) fn install(
    limit: usize,
    fault: Fault,
    erased: impl FnOnce() + 'static,
) -> Session {
    STATE.with(|slot| {
        let mut slot = slot.borrow_mut();
        assert!(slot.is_none(), "nested restoration fault session");
        *slot = Some(State {
            limit,
            fault,
            extractions: 0,
            reads: 0,
            erased: Some(Box::new(erased)),
        });
    });
    Session
}

impl Drop for Session {
    fn drop(&mut self) {
        // Release the RefCell borrow before dropping captured observers.
        let state = STATE.with(|slot| slot.borrow_mut().take());
        drop(state);
    }
}

pub(in crate::replicated_decoder) fn limit() -> usize {
    STATE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(MAX_PROGRAM_BYTES, |s| s.limit)
    })
}

pub(in crate::replicated_decoder) fn extracting() {
    STATE.with(|slot| {
        if let Some(state) = slot.borrow_mut().as_mut() {
            state.extractions += 1;
        }
    });
}

pub(in crate::replicated_decoder) fn counts() -> (usize, usize) {
    STATE.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or((0, 0), |s| (s.extractions, s.reads))
    })
}

pub(in crate::replicated_decoder) fn reading() -> Result<(), Error> {
    let fault = STATE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(state) = slot.as_mut() else {
            return Fault::None;
        };
        state.reads += 1;
        if state.reads == 2 {
            state.fault
        } else {
            Fault::None
        }
    });
    // No RefCell borrow spans the reader failure/unwind or output destruction.
    match fault {
        Fault::None => Ok(()),
        Fault::ReadError => Err(Error::Binding),
        Fault::ReadPanic => panic!("injected restoration reader unwind"),
    }
}

// This field follows Values in the actual caller's output container. Rust drops
// it only AFTER the entire native Values destructor has completed. Observers
// inspect Weak owner counts, never erased memory or native runtime internals.
pub(in crate::replicated_decoder) struct ErasureProbe(Option<Box<dyn FnOnce()>>);

impl ErasureProbe {
    pub(in crate::replicated_decoder) fn take() -> Self {
        Self(STATE.with(|slot| slot.borrow_mut().as_mut().and_then(|s| s.erased.take())))
    }
}

impl Drop for ErasureProbe {
    fn drop(&mut self) {
        if let Some(observe) = self.0.take() {
            observe();
        }
    }
}

#[test]
fn owned_restoration_controls_are_additive_and_preflight_keeps_original_cap() {
    let guard = crate::replicated_decoder::runtime::body::handoff::owned_restore_inline_bytes::<
        super::RestoreOutputs,
    >()
    .unwrap();
    let controls = super::owned_restoration_controls().unwrap();
    assert!(controls > guard + std::mem::size_of::<super::RestoreOutputs>());
    let original = 137usize;
    let exact = original.checked_add(controls).unwrap();
    assert_eq!(super::restoration_preflight(exact, exact), Ok(()));
    assert_eq!(
        super::restoration_preflight(exact, exact - 1),
        Err(Error::Bounds)
    );
    assert_eq!(
        super::restoration_preflight(exact, original),
        Err(Error::Bounds)
    );
    assert_eq!(
        super::restoration_preflight(MAX_PROGRAM_BYTES + 1, usize::MAX),
        Err(Error::Bounds)
    );
}
