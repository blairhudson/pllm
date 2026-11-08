use super::*;

// Borrows only the state field, never the Program or pending allocation. There
// is no rollback: ordinary Err and unwinding both leave original owners resident.
struct Entered<'a> {
    state: &'a mut ResidentState,
}

impl Drop for Entered<'_> {
    fn drop(&mut self) {
        if *self.state == ResidentState::Entering {
            *self.state = ResidentState::Failed;
        }
    }
}

impl ResidentBuilder {
    pub(crate) fn from_builder(builder: Builder) -> Self {
        Self {
            builder: Some(builder),
            pending_affine: ExactStorage::empty(),
            state: ResidentState::Ready,
        }
    }

    pub(crate) fn affine_lanes(
        &mut self,
        input: Value,
        coefficients: &[Affine],
    ) -> Result<Value, Error> {
        let Self {
            builder,
            pending_affine,
            state,
        } = self;
        if *state != ResidentState::Ready {
            return Err(Error::Phase);
        }
        *state = ResidentState::Entering;
        let entered = Entered { state };
        let output = builder
            .as_mut()
            .ok_or(Error::Phase)?
            .affine_lanes_with_pending(input, coefficients, pending_affine)?;
        *entered.state = ResidentState::Ready;
        Ok(output)
    }

    pub(crate) fn finish(&mut self, output: Value) -> Result<(), Error> {
        if !matches!(self.pending_affine, ExactStorage::Empty) { return Err(Error::Phase); }
        let Self {
            builder,
            pending_affine: _,
            state,
        } = self;
        if *state != ResidentState::Ready {
            return Err(Error::Phase);
        }
        *state = ResidentState::Entering;
        let entered = Entered { state };
        builder.as_mut().ok_or(Error::Phase)?.finish_in_place(output)?;
        *entered.state = ResidentState::Complete;
        Ok(())
    }

    pub(crate) fn finish_checked(&mut self, output: Value, valid: Value) -> Result<(), Error> {
        if !matches!(self.pending_affine, ExactStorage::Empty) { return Err(Error::Phase); }
        let Self {
            builder,
            pending_affine: _,
            state,
        } = self;
        if *state != ResidentState::Ready {
            return Err(Error::Phase);
        }
        *state = ResidentState::Entering;
        let entered = Entered { state };
        builder
            .as_mut()
            .ok_or(Error::Phase)?
            .finish_checked_in_place(output, valid)?;
        *entered.state = ResidentState::Complete;
        Ok(())
    }

    /// Install once into a caller-resident empty slot. Refusal changes nothing.
    pub(crate) fn transfer_complete(&mut self, out: &mut Option<Program>) -> Result<(), Error> {
        if out.is_some() {
            return Err(Error::Phase);
        }
        if self.state != ResidentState::Complete {
            return Err(Error::Phase);
        }
        if self.builder.is_none() {
            return Err(Error::Phase);
        }
        *out = self.builder.take().map(|Builder(program)| program);
        self.state = ResidentState::Transferred;
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    static PANIC_AFTER_AFFINE_RESERVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static RESERVED_AFFINE: std::cell::Cell<(*const Affine, usize)> = const {
        std::cell::Cell::new((std::ptr::null(), 0))
    };
    static PANIC_AFTER_PARTIAL_SCHEDULE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static PANIC_ON_FINISH: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static FINISH_HOOK_FIRES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FINISH_HOOK_PANICS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FAIL_AFFINE_RESERVE: std::cell::Cell<Option<(usize, bool)>> = const { std::cell::Cell::new(None) };
    static AFFINE_INITIALIZE_INVOCATION: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FAIL_AFFINE_INITIALIZE: std::cell::Cell<Option<(usize, usize, bool)>> = const { std::cell::Cell::new(None) };
    static INITIALIZED_AFFINE: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
    static READY_BEFORE_AFFINE_INITIALIZE: std::cell::Cell<Option<(*const Affine, usize)>> = const { std::cell::Cell::new(None) };
}

/// Test-only fault after a selected write, while backing is still Building.
/// This observes extent/prefix, never the shared allocator's opaque pointer.
#[cfg(test)]
pub(super) fn after_affine_initialize(table: &ExactStorage<Affine>) -> Result<(), Error> {
    let invocation = AFFINE_INITIALIZE_INVOCATION.with(|count| {
        if table.len() == 1 { count.set(count.get() + 1); }
        count.get()
    });
    FAIL_AFFINE_INITIALIZE.with(|fault| {
        if let Some((selected, prefix, unwind)) = fault.get() {
            if selected == invocation && table.len() == prefix {
                assert!(matches!(table, ExactStorage::Building { .. }));
                fault.set(None);
                INITIALIZED_AFFINE.with(|observed| observed.set((table.extent(), table.len())));
                if unwind { panic!("test resident affine initialized prefix before seal"); }
                return Err(Error::Allocation);
            }
        }
        Ok(())
    })
}

#[cfg(test)]
pub(super) fn arm_affine_initialize_failure(invocation: usize, prefix: usize, unwind: bool) {
    assert!(invocation > 0 && prefix > 0);
    let selected = AFFINE_INITIALIZE_INVOCATION.with(|count| count.get() + invocation);
    FAIL_AFFINE_INITIALIZE.with(|fault| assert!(fault.replace(Some((selected, prefix, unwind))).is_none()));
    READY_BEFORE_AFFINE_INITIALIZE.with(|observed| observed.set(None));
}

#[cfg(test)]
pub(super) fn initialized_affine() -> (usize, usize) {
    INITIALIZED_AFFINE.with(|observed| observed.get())
}

#[cfg(test)]
pub(super) fn ready_before_affine_initialize() -> Option<(*const Affine, usize)> {
    READY_BEFORE_AFFINE_INITIALIZE.with(|observed| observed.get())
}

#[cfg(test)]
pub(super) fn after_affine_reserve(table: &ExactStorage<Affine>) -> Result<(), Error> {
    super::rms_resident::observe_affine(table.ready()?);
    if FAIL_AFFINE_INITIALIZE.with(|fault| fault.get().is_some()) {
        READY_BEFORE_AFFINE_INITIALIZE.with(|observed| observed.set(Some((table.as_ptr(), table.extent()))));
    }
    PANIC_AFTER_AFFINE_RESERVE.with(|flag| {
        if flag.replace(false) {
            RESERVED_AFFINE.with(|observed| observed.set((table.as_ptr(), table.extent())));
            panic!("test resident affine after seal before handoff");
        }
    });
    FAIL_AFFINE_RESERVE.with(|fault| {
        if let Some((remaining, unwind)) = fault.get() {
            if remaining == 1 {
                fault.set(None);
                RESERVED_AFFINE.with(|observed| observed.set((table.as_ptr(), table.extent())));
                if unwind { panic!("test resident affine acquisition unwind"); }
                return Err(Error::Allocation);
            }
            fault.set(Some((remaining - 1, unwind)));
        }
        Ok(())
    })
}

#[cfg(test)]
pub(super) fn arm_affine_reserve_failure(invocation: usize, unwind: bool) {
    assert!(invocation > 0);
    FAIL_AFFINE_RESERVE.with(|fault| assert!(fault.replace(Some((invocation, unwind))).is_none()));
}

#[cfg(test)]
pub(super) fn after_schedule_slot(index: usize) {
    if index == 1 {
        FINISH_HOOK_FIRES.with(|count| count.set(count.get() + 1));
        PANIC_ON_FINISH.with(|pending| {
            if let Some(remaining) = pending.get() {
                if remaining == 1 {
                    pending.set(None);
                    FINISH_HOOK_PANICS.with(|count| count.set(count.get() + 1));
                    panic!("test rotary resident finish after partial schedule");
                }
                pending.set(Some(remaining - 1));
            }
        });
        PANIC_AFTER_PARTIAL_SCHEDULE.with(|flag| {
            if flag.replace(false) {
                panic!("test resident finish after partial schedule");
            }
        });
    }
}

#[cfg(test)]
pub(super) fn arm_finish_panic(invocation: usize) {
    assert!(invocation > 0);
    PANIC_ON_FINISH.with(|pending| assert!(pending.replace(Some(invocation)).is_none()));
}

#[cfg(test)]
pub(super) fn finish_hook_counts() -> (usize, usize) {
    (FINISH_HOOK_FIRES.with(|count| count.get()), FINISH_HOOK_PANICS.with(|count| count.get()))
}

#[cfg(test)]
pub(super) fn arm_affine_reserve_panic() {
    PANIC_AFTER_AFFINE_RESERVE.with(|flag| assert!(!flag.replace(true)));
}

#[cfg(test)]
pub(super) fn reserved_affine() -> (*const Affine, usize) {
    RESERVED_AFFINE.with(|observed| observed.get())
}

#[cfg(test)]
#[path = "tensor_resident_builder_tests.rs"]
mod tests;
