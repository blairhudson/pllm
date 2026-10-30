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
            pending_affine: Vec::new(),
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
}

#[cfg(test)]
pub(super) fn after_affine_reserve(table: &Vec<Affine>) {
    PANIC_AFTER_AFFINE_RESERVE.with(|flag| {
        if flag.replace(false) {
            RESERVED_AFFINE.with(|observed| observed.set((table.as_ptr(), table.capacity())));
            panic!("test resident affine after reserve before capacity check");
        }
    });
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
#[path = "tensor_resident_builder_tests.rs"]
mod tests;
