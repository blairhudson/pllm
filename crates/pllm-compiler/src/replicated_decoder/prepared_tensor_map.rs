//! Closed, compiler-owned preparation for ordinary flattened tensor maps.
//!
//! No constructor wraps an arbitrary Program or accepts an external row proof.
//! The captured interval is a local compiler-declaration subset contract, not
//! external proof or checkpoint/source authority.
use super::{puma, tensor, Error, Interval, WorkBudget, MAX_VALUES};
use sha2::{Digest, Sha256};
use std::{mem::size_of, sync::Arc};

#[cfg(test)]
thread_local! {
    static NATIVE_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    // Opt-in test observation only. Weak handles retain allocation backing, so
    // never accumulate them during ordinary construction or unrelated tests.
    static NATIVE_OWNERS: std::cell::RefCell<Option<Vec<std::sync::Weak<tensor::Program>>>> = const { std::cell::RefCell::new(None) };
}
#[cfg(test)]
pub(super) fn native_attempts() -> usize {
    NATIVE_ATTEMPTS.with(|n| n.get())
}

#[cfg(test)]
pub(super) fn observe_native_owners() {
    NATIVE_OWNERS.with(|owners| {
        assert!(owners.borrow().is_none());
        *owners.borrow_mut() = Some(Vec::new());
    });
}

#[cfg(test)]
pub(super) fn take_native_owners() -> Vec<std::sync::Weak<tensor::Program>> {
    NATIVE_OWNERS.with(|owners| owners.borrow_mut().take().unwrap_or_default())
}

pub(super) struct PreparedTensorMap {
    lanes: usize,
    input: Interval,
    factory: Factory,
    full: Arc<tensor::Program>,
    tail: Option<Arc<tensor::Program>>,
}

/// Read-only scope for a future one-loop bridge, not a proof or routing decision.
/// Only compiler-owned preparation can supply this classification; program
/// digests and independently allocated templates cannot establish provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OneLoopScope {
    Rescale18,
    GuardedSilu,
}

#[derive(Clone, Copy)]
enum Factory {
    Rescale(u8),
    GuardedSilu,
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}

type Programs<'a> = std::iter::Chain<
    std::iter::Once<&'a Arc<tensor::Program>>,
    std::option::Iter<'a, Arc<tensor::Program>>,
>;

// Logical named inline owners/move positions, not allocator/RSS/ABI stack.
// Native Program/Builder buffers, its Program local and Arc publication are in
// ConstructionQuote. The extra Factory::make bridge Result is priced here.
// Scalar arithmetic/quote-calculation temporaries and std implementation stack
// are not owned buffers. No padding allowance substitutes for a named slot.
struct AssemblyInventory {
    quote: usize,
    full: usize,
    tail: usize,
    owner: usize,
}

impl AssemblyInventory {
    fn for_callback<F>() -> Result<Self, Error> {
        // Checked scalar sums without introducing an array/iterator owner.
        macro_rules! slots {
            ($($bytes:expr),+ $(,)?) => {{
                let mut total = 0;
                $(total = add(total, $bytes)?;)+
                total
            }};
        }
        let common = slots!(
            size_of::<F>(), // public constructor parameter -> prepare parameter
            size_of::<F>(),
            size_of::<&mut F>(), // real FnMut receiver, not its pointee/heap
            size_of::<Result<(), Error>>(), // admission callback return
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<tensor::ConstructionQuote>(), // full_quote
            size_of::<Option<tensor::ConstructionQuote>>(), // tail_quote
            size_of::<Result<PreparedTensorMap, Error>>(), // prepare return
            size_of::<Result<PreparedTensorMap, Error>>(), // wrapper return move
        );
        let factory_quote = slots!(
            size_of::<tensor::Construction>(), // Factory::quote local/receiver
            size_of::<tensor::Construction>(),
            size_of::<Result<tensor::ConstructionQuote, tensor::ConstructionQuoteError>>(),
            size_of::<Result<tensor::ConstructionQuote, Error>>(), // map_err
            size_of::<Result<tensor::ConstructionQuote, Error>>(), // bridge return
            size_of::<Option<tensor::ConstructionQuote>>(),        // Some expression
        );
        // The later if-let used to combine peak phases has a Quote binding but
        // no live Factory::quote bridge. Take the larger named source phase.
        let quote = add(
            common,
            factory_quote.max(size_of::<tensor::ConstructionQuote>()),
        )?;
        let full = slots!(
            common,
            size_of::<Arc<tensor::Program>>(), // full destination/move
            size_of::<Arc<tensor::Program>>(), // make_admitted owner -> return
            size_of::<tensor::ConstructionQuote>(), // make_admitted argument
            size_of::<Result<tensor::Program, Error>>(), // Factory::make bridge
            size_of::<Result<Arc<tensor::Program>, Error>>(), // make_admitted return
            size_of::<tensor::Resources>(),
            size_of::<Result<tensor::Resources, Error>>(),
        );
        let tail = slots!(
            full,
            size_of::<tensor::ConstructionQuote>(), // if-let quote binding
            size_of::<Option<Arc<tensor::Program>>>(), // Some expression
            size_of::<Option<Arc<tensor::Program>>>(), // tail binding/move
        );
        // programs(): Once/option-Iter expression owners, chain expression,
        // returned chain, consumer iterator. No template/backing alias credit.
        let visit = slots!(
            size_of::<std::iter::Once<&Arc<tensor::Program>>>(),
            size_of::<std::option::Iter<'_, Arc<tensor::Program>>>(),
            size_of::<Programs<'_>>(),
            size_of::<Programs<'_>>(),
            size_of::<Programs<'_>>(),
            size_of::<&Arc<tensor::Program>>(),
            // validity_lanes' Option is consumed before the resource census.
            slots!(
                size_of::<tensor::Resources>(),
                size_of::<Result<tensor::Resources, Error>>()
            )
            .max(size_of::<Option<usize>>()),
        );
        let geometry = slots!(
            size_of::<Option<&Arc<tensor::Program>>>(), // tail.as_ref result
            size_of::<Option<&Arc<tensor::Program>>>(), // is_some_and receiver
            size_of::<&usize>(),                        // remainder predicate capture
            size_of::<&Arc<tensor::Program>>(),         // predicate argument
            size_of::<Result<(), Error>>(),
        );
        let validation = geometry.max(slots!(visit, size_of::<Result<(), Error>>()));
        let census = slots!(
            visit,
            size_of::<&mut Programs<'_>>(),    // try_fold receiver
            size_of::<Result<usize, Error>>(), // try_fold return
            size_of::<Result<usize, Error>>(), // template_bytes return
        );
        let owner = slots!(
            common,
            size_of::<PreparedTensorMap>(),    // assembled owner
            size_of::<Arc<tensor::Program>>(), // full source move
            size_of::<Option<Arc<tensor::Program>>>(), // tail source move
            validation.max(census),            // sequential visits, not simultaneous ones
        );
        Ok(Self {
            quote,
            full,
            tail,
            owner,
        })
    }
}

impl Factory {
    fn quote(self, lanes: usize) -> Result<tensor::ConstructionQuote, Error> {
        let construction = match self {
            Self::Rescale(_) => tensor::Construction::Builder { lanes },
            Self::GuardedSilu => tensor::Construction::GuardedSilu { lanes },
        };
        construction.quote().map_err(|e| match e {
            tensor::ConstructionQuoteError::Geometry(e) => e,
            tensor::ConstructionQuoteError::UncheckedRotaryCoefficientCapacity => Error::Bounds,
        })
    }

    fn make(self, lanes: usize, input: Interval) -> Result<tensor::Program, Error> {
        match self {
            Self::Rescale(shift) => {
                let mut b = tensor::Builder::new(lanes, input)?;
                let output = b.rescale(b.input(), shift)?;
                b.finish(output)
            }
            Self::GuardedSilu => {
                tensor::Program::activation_guarded(tensor::Activation::Silu, lanes, input)
            }
        }
    }
}

impl PreparedTensorMap {
    pub(super) fn rescale<F: FnMut(usize) -> Result<(), Error>>(
        lanes: usize,
        input: Interval,
        shift: u8,
        caller_baseline: usize,
        admit: F,
    ) -> Result<Self, Error> {
        Self::prepare(
            lanes,
            input,
            Factory::Rescale(shift),
            caller_baseline,
            admit,
        )
    }

    pub(super) fn guarded_silu<F: FnMut(usize) -> Result<(), Error>>(
        lanes: usize,
        input: Interval,
        caller_baseline: usize,
        admit: F,
    ) -> Result<Self, Error> {
        Self::prepare(lanes, input, Factory::GuardedSilu, caller_baseline, admit)
    }

    // F's two inline move positions are priced, not arbitrary callback work or
    // its owned heap/borrowed pointees. Caller baseline + the REAL admission
    // callback must account those independent owners (including a borrowed
    // callback object's own inline storage). This is not a Budget, source proof
    // or complete-constructor/request envelope.
    fn prepare<F: FnMut(usize) -> Result<(), Error>>(
        lanes: usize,
        input: Interval,
        factory: Factory,
        caller_baseline: usize,
        mut admit: F,
    ) -> Result<Self, Error> {
        if lanes == 0 || lanes > MAX_VALUES {
            return Err(Error::Bounds);
        }
        let width = lanes.min(tensor::MAX_TILE_LANES);
        let remainder = lanes % width;
        let full_quote = factory.quote(width)?;
        let tail_quote = if remainder != 0 {
            Some(factory.quote(remainder)?)
        } else {
            None
        };
        let assembly = AssemblyInventory::for_callback::<F>()?;
        let mut peak = assembly
            .quote
            .max(add(assembly.full, full_quote.new_peak_bytes)?);
        let mut templates = full_quote.immutable_template_bytes;
        if let Some(quote) = tail_quote {
            peak = peak.max(add(
                assembly.tail,
                add(full_quote.immutable_template_bytes, quote.new_peak_bytes)?,
            )?);
            templates = add(templates, quote.immutable_template_bytes)?;
        }
        peak = add(caller_baseline, peak.max(add(assembly.owner, templates)?))?;
        // Admit BOTH constructions before the first native factory can allocate.
        admit(peak)?;
        // Preserve the original first-callback error/unwind precedence, but do
        // not enter a numeric factory with its known-invalid shift parameter.
        if matches!(factory, Factory::Rescale(shift) if !(1..=62).contains(&shift)) {
            return Err(Error::Bounds);
        }
        let full = Self::make_admitted(factory, width, input, full_quote)?;
        let tail = if let Some(quote) = tail_quote {
            // Repeat with actual retained capacity, not just constructor's bound.
            admit(add(
                caller_baseline,
                add(
                    assembly.tail,
                    add(
                        full.resources()?.immutable_template_bytes,
                        quote.new_peak_bytes,
                    )?,
                )?,
            )?)?;
            Some(Self::make_admitted(factory, remainder, input, quote)?)
        } else {
            None
        };
        let prepared = Self {
            lanes,
            input,
            factory,
            full,
            tail,
        };
        prepared.validate()?;
        admit(add(
            caller_baseline,
            add(assembly.owner, prepared.template_bytes()?)?,
        )?)?;
        Ok(prepared)
    }

    fn make_admitted(
        factory: Factory,
        lanes: usize,
        input: Interval,
        quote: tensor::ConstructionQuote,
    ) -> Result<Arc<tensor::Program>, Error> {
        #[cfg(test)]
        NATIVE_ATTEMPTS.with(|n| n.set(n.get() + 1));
        let program = factory.make(lanes, input)?;
        let resources = program.resources()?;
        if resources.immutable_template_bytes > quote.immutable_template_bytes
            || resources.shared_admitted_bytes > puma::MAX_ADMITTED_BYTES
        {
            return Err(Error::Bounds);
        }
        let owner = Arc::new(program);
        #[cfg(test)]
        NATIVE_OWNERS.with(|owners| {
            if let Some(owners) = &mut *owners.borrow_mut() {
                owners.push(Arc::downgrade(&owner));
            }
        });
        Ok(owner)
    }

    pub(super) fn validate(&self) -> Result<(), Error> {
        if self.lanes == 0 || self.lanes > MAX_VALUES {
            return Err(Error::Bounds);
        }
        let width = self.lanes.min(tensor::MAX_TILE_LANES);
        let remainder = self.lanes % width;
        if self.full.lanes() != width
            || self.tail.is_some() != (remainder != 0)
            || self.tail.as_ref().is_some_and(|p| p.lanes() != remainder)
        {
            return Err(Error::Shape);
        }
        for program in self.programs() {
            if program.output_lanes() != program.lanes()
                || program.validity_lanes().is_some()
                || program.instructions() == 0
            {
                return Err(Error::Shape);
            }
            if program.resources()?.shared_admitted_bytes > puma::MAX_ADMITTED_BYTES {
                return Err(Error::Bounds);
            }
        }
        Ok(())
    }

    /// Compare local compiler declarations only. Acceptance neither verifies
    /// secret values nor certifies an external source or its declared interval.
    pub(super) fn accepts(&self, lanes: usize, actual: Interval) -> bool {
        lanes == self.lanes
            && actual.lower() >= self.input.lower()
            && actual.upper() <= self.input.upper()
    }

    /// Classify the captured factory without inferring scope from templates or
    /// changing their original full/tail Arc owners. Other rescale shifts remain
    /// usable by existing sinks, but are outside the initial one-loop scope.
    #[allow(dead_code)]
    pub(super) fn one_loop_scope(&self) -> Option<OneLoopScope> {
        match self.factory {
            Factory::Rescale(18) => Some(OneLoopScope::Rescale18),
            Factory::GuardedSilu => Some(OneLoopScope::GuardedSilu),
            Factory::Rescale(_) => None,
        }
    }

    pub(super) fn lanes(&self) -> usize {
        self.lanes
    }

    pub(super) fn programs(&self) -> Programs<'_> {
        std::iter::once(&self.full).chain(self.tail.iter())
    }

    pub(super) fn program_at(&self, offset: usize) -> Result<&Arc<tensor::Program>, Error> {
        let width = self.full.lanes();
        if offset >= self.lanes || offset % width != 0 {
            return Err(Error::Shape);
        }
        if self.lanes - offset < width {
            self.tail.as_ref().ok_or(Error::Shape)
        } else {
            Ok(&self.full)
        }
    }

    pub(super) fn output_bounds(&self) -> Result<Interval, Error> {
        let full = self.full.output_bounds();
        let tail = self.tail.as_ref().map_or(full, |p| p.output_bounds());
        Interval::new(
            full.lower().min(tail.lower()),
            full.upper().max(tail.upper()),
        )
    }

    pub(super) fn template_bytes(&self) -> Result<usize, Error> {
        self.programs()
            .try_fold(0, |n, p| add(n, p.resources()?.immutable_template_bytes))
    }

    /// Pre-existing caller value + its actual Arc allocations/table capacities.
    /// Enclosing owners must retain this baseline until that caller drops it.
    pub(super) fn baseline_bytes(&self) -> Result<usize, Error> {
        add(size_of::<Self>(), self.template_bytes()?)
    }

    pub(super) fn work_budget(&self) -> Result<WorkBudget, Error> {
        let full = self.lanes / self.full.lanes();
        let work = self.full.work_budget().checked_mul(full as u64)?;
        work.checked_add(
            self.tail
                .as_ref()
                .map_or(WorkBudget::default(), |p| p.work_budget()),
        )
    }

    pub(super) fn advances(&self) -> Result<u64, Error> {
        (self.full.instructions() as u64)
            .checked_mul((self.lanes / self.full.lanes()) as u64)
            .and_then(|n| n.checked_add(self.tail.as_ref().map_or(0, |p| p.instructions() as u64)))
            .ok_or(Error::Bounds)
    }

    pub(super) fn invocations(&self) -> u64 {
        self.lanes.div_ceil(self.full.lanes()) as u64
    }

    pub(super) fn hash(&self, hash: &mut Sha256) {
        hash.update((self.lanes as u64).to_le_bytes());
        hash.update(self.input.lower().to_le_bytes());
        hash.update(self.input.upper().to_le_bytes());
        hash.update(self.full.digest());
        hash.update([u8::from(self.tail.is_some())]);
        if let Some(tail) = &self.tail {
            hash.update(tail.digest());
        }
    }
}
