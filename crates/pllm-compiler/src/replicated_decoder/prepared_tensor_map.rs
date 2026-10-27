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

// Local construction quote, not an execution/work permit. Native quotes remain
// Construction::quote values; assembly names separate construction phases.
struct MapInventory {
    width: usize,
    remainder: usize,
    full_quote: tensor::ConstructionQuote,
    tail_quote: Option<tensor::ConstructionQuote>,
    assembly: AssemblyInventory,
    peak: usize,
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

fn construction_inventory<F, G: WorkGate>(
    lanes: usize,
    factory: Factory,
    caller_baseline: usize,
) -> Result<MapInventory, Error> {
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
    let mut assembly = AssemblyInventory::for_callback::<F>()?;
    // Helper return, Result bridge and prepare binding are named logical move
    // positions, not a physical-stack/allocator claim.
    let inventory_moves = add(
        size_of::<MapInventory>()
            .checked_mul(2)
            .ok_or(Error::Bounds)?,
        size_of::<Result<MapInventory, Error>>(),
    )?;
    assembly.quote = add(assembly.quote, inventory_moves)?;
    assembly.full = add(assembly.full, inventory_moves)?;
    assembly.tail = add(assembly.tail, inventory_moves)?;
    assembly.owner = add(assembly.owner, inventory_moves)?;
    if matches!(factory, Factory::Rescale(_)) {
        // New Program::rescale return -> existing Factory::make return bridge.
        // Keep the native Construction::quote unchanged; price this wrapper here.
        let native_bridge = size_of::<Result<tensor::Program, Error>>();
        assembly.full = add(assembly.full, native_bridge)?;
        assembly.tail = add(assembly.tail, native_bridge)?;
    }
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
    peak = peak.max(add(assembly.owner, templates)?);
    peak = peak.max(add(assembly.quote, G::workspace_phase_bytes()?)?);
    // Original prior argument, gate expression, prepare parameter and receiver.
    peak = add(
        add(caller_baseline, peak)?,
        size_of::<G>().checked_mul(4).ok_or(Error::Bounds)?,
    )?;
    Ok(MapInventory {
        width,
        remainder,
        full_quote,
        tail_quote,
        assembly,
        peak,
    })
}

// Caller supplies already-admitted workspace; neither branch constructs it or
// enters a native factory. Rescale's existing NoWorkGate remains unchanged.
enum MapWorkWorkspace<'a> {
    #[cfg_attr(not(test), allow(dead_code))]
    Rescale(&'a mut tensor::SoftmaxWorkWorkspace),
    GuardedSilu(&'a mut tensor::ActivationWorkWorkspace),
}

fn scheduled_work(
    factory: Factory,
    lanes: usize,
    input: Interval,
    mut workspace: MapWorkWorkspace<'_>,
) -> Result<WorkBudget, Error> {
    if lanes == 0 || lanes > MAX_VALUES {
        return Err(Error::Bounds);
    }
    let width = lanes.min(tensor::MAX_TILE_LANES);
    let remainder = lanes % width;
    let mut scheduled = factory.query_work(width, input, &mut workspace)?
        .checked_mul(u64::try_from(lanes / width).map_err(|_| Error::Bounds)?)?;
    if remainder != 0 {
        scheduled = scheduled.checked_add(factory.query_work(remainder, input, &mut workspace)?)?;
    }
    Ok(scheduled)
}

impl Factory {
    fn query_work(
        self,
        lanes: usize,
        input: Interval,
        workspace: &mut MapWorkWorkspace<'_>,
    ) -> Result<WorkBudget, Error> {
        match (self, workspace) {
            (Self::Rescale(shift), MapWorkWorkspace::Rescale(workspace)) => {
                tensor::Program::rescale_work(lanes, input, shift, workspace)
            }
            (Self::GuardedSilu, MapWorkWorkspace::GuardedSilu(workspace)) => {
                tensor::Program::activation_guarded_work(
                    tensor::Activation::Silu,
                    lanes,
                    input,
                    workspace,
                )
            }
            _ => Err(Error::Shape),
        }
    }

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
            Self::Rescale(shift) => tensor::Program::rescale(lanes, input, shift),
            Self::GuardedSilu => {
                tensor::Program::activation_guarded(tensor::Activation::Silu, lanes, input)
            }
        }
    }
}

// Private, closed pre-factory gate; not a Budget, request permit or work debit.
// Zero-sized on the unchanged rescale path. Guarded activation carries the
// caller's ACTUAL already-published work, never resets it to a local subtotal.
trait WorkGate {
    fn workspace_phase_bytes() -> Result<usize, Error>;
    fn check(
        self,
        lanes: usize,
        width: usize,
        remainder: usize,
        input: Interval,
    ) -> Result<(), Error>;
}
struct NoWorkGate;
impl WorkGate for NoWorkGate {
    fn workspace_phase_bytes() -> Result<usize, Error> {
        Ok(0)
    }
    fn check(self, _: usize, _: usize, _: usize, _: Interval) -> Result<(), Error> {
        Ok(())
    }
}
struct ActivationWorkGate {
    prior: WorkBudget,
}
impl WorkGate for ActivationWorkGate {
    fn workspace_phase_bytes() -> Result<usize, Error> {
        // Workspace plus named native-query/aggregation return and move slots.
        // No native Program here; native construction is a later, disjoint phase.
        // empty() return and caller-owned binding, even when a move is elided.
        let workspace = tensor::ActivationWorkWorkspace::workspace_bytes()
            .checked_mul(2)
            .ok_or(Error::Bounds)?;
        let values = size_of::<WorkBudget>()
            // full query/mul receiver; scheduled_work; tail; scheduled-add
            // receiver/argument; final-add receiver/argument; total; limits receiver
            .checked_mul(10) // includes scheduled_work's returned aggregate
            .ok_or(Error::Bounds)?;
        let returns = size_of::<Result<WorkBudget, Error>>()
            .checked_mul(5) // native/mul/add, query_work and scheduled_work bridges
            .ok_or(Error::Bounds)?;
        let scheduled_query = add(
            size_of::<MapWorkWorkspace<'_>>()
                .checked_mul(2) // argument expression and helper parameter
                .ok_or(Error::Bounds)?,
            size_of::<&mut MapWorkWorkspace<'_>>(), // query_work parameter
        )?;
        add(
            add(add(workspace, values)?, returns)?,
            add(
                scheduled_query,
                add(
                    size_of::<Result<puma::prss::PrssLimits, Error>>(),
                    size_of::<Result<(), Error>>(),
                )?,
            )?,
        )
    }
    fn check(
        self,
        lanes: usize,
        _: usize,
        _: usize,
        input: Interval,
    ) -> Result<(), Error> {
        let mut workspace = tensor::ActivationWorkWorkspace::empty();
        let scheduled = scheduled_work(
            Factory::GuardedSilu,
            lanes,
            input,
            MapWorkWorkspace::GuardedSilu(&mut workspace),
        )?;
        let total = self.prior.checked_add(scheduled)?;
        total.limits()?;
        Ok(())
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
            NoWorkGate,
        )
    }

    // Standalone/local factory seam retained for other compiler emitters. BODY
    // supplies actual prior work through guarded_silu_with_prior_work below.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn guarded_silu<F: FnMut(usize) -> Result<(), Error>>(
        lanes: usize,
        input: Interval,
        caller_baseline: usize,
        admit: F,
    ) -> Result<Self, Error> {
        // Standalone factory users have no caller work supplied: this admits
        // ONLY the local schedule. It cannot stand in for global admission.
        Self::prepare(
            lanes,
            input,
            Factory::GuardedSilu,
            caller_baseline,
            admit,
            ActivationWorkGate {
                prior: WorkBudget::default(),
            },
        )
    }

    pub(super) fn guarded_silu_with_prior_work<F: FnMut(usize) -> Result<(), Error>>(
        lanes: usize,
        input: Interval,
        caller_baseline: usize,
        prior_work: WorkBudget,
        admit: F,
    ) -> Result<Self, Error> {
        Self::prepare(
            lanes,
            input,
            Factory::GuardedSilu,
            caller_baseline,
            admit,
            ActivationWorkGate { prior: prior_work },
        )
    }

    // F's two inline move positions are priced, not arbitrary callback work or
    // its owned heap/borrowed pointees. Caller baseline + the REAL admission
    // callback must account those independent owners (including a borrowed
    // callback object's own inline storage). This is not a Budget, source proof
    // or complete-constructor/request envelope.
    fn prepare<F: FnMut(usize) -> Result<(), Error>, G: WorkGate>(
        lanes: usize,
        input: Interval,
        factory: Factory,
        caller_baseline: usize,
        mut admit: F,
        work_gate: G,
    ) -> Result<Self, Error> {
        let inventory = construction_inventory::<F, G>(lanes, factory, caller_baseline)?;
        // Admit BOTH constructions before the first native factory can allocate.
        admit(inventory.peak)?;
        // Preserve the original first-callback error/unwind precedence, but do
        // not enter a numeric factory with its known-invalid shift parameter.
        if matches!(factory, Factory::Rescale(shift) if !(1..=62).contains(&shift)) {
            return Err(Error::Bounds);
        }
        work_gate.check(lanes, inventory.width, inventory.remainder, input)?;
        let full = Self::make_admitted(factory, inventory.width, input, inventory.full_quote)?;
        let tail = if let Some(quote) = inventory.tail_quote {
            // Repeat with actual retained capacity, not just constructor's bound.
            admit(add(
                caller_baseline,
                add(
                    inventory.assembly.tail,
                    add(
                        full.resources()?.immutable_template_bytes,
                        quote.new_peak_bytes,
                    )?,
                )?,
            )?)?;
            Some(Self::make_admitted(factory, inventory.remainder, input, quote)?)
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
            add(inventory.assembly.owner, prepared.template_bytes()?)?,
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

#[cfg(test)]
#[path = "prepared_activation_work_tests.rs"]
mod activation_work_tests;

#[path = "prepared_tensor_map/prepared_tensor_request_rescale.rs"]
pub(in crate::replicated_decoder) mod request_rescale;

#[path = "prepared_tensor_map/prepared_tensor_request_silu.rs"]
pub(in crate::replicated_decoder) mod request_silu;

#[cfg(test)]
mod stage_a_tests {
    use super::*;

    const ONE: i64 = 1 << 18;

    // Literal old recipe, independent of Program::rescale and both queries.
    fn old_rescale(lanes: usize, input: Interval, shift: u8) -> tensor::Program {
        let mut builder = tensor::Builder::new(lanes, input).unwrap();
        let output = builder.rescale(builder.input(), shift).unwrap();
        builder.finish(output).unwrap()
    }

    #[test]
    fn scheduled_queries_match_full_tail_native_work_without_map_factories() {
        let input = Interval::new(-ONE, ONE).unwrap();
        // Tests supply workspace explicitly; this is not an admission witness.
        let mut rescale = tensor::SoftmaxWorkWorkspace::empty();
        let mut activation = tensor::ActivationWorkWorkspace::empty();
        for lanes in [1, 3, 127, 128, 129, 256, 257, 260] {
            let before = native_attempts();
            observe_native_owners();
            let rescale_work = scheduled_work(
                Factory::Rescale(18),
                lanes,
                input,
                MapWorkWorkspace::Rescale(&mut rescale),
            )
            .unwrap();
            let activation_work = scheduled_work(
                Factory::GuardedSilu,
                lanes,
                input,
                MapWorkWorkspace::GuardedSilu(&mut activation),
            )
            .unwrap();
            assert_eq!(native_attempts(), before);
            assert!(take_native_owners().is_empty());
            let width = lanes.min(tensor::MAX_TILE_LANES);
            let remainder = lanes % width;
            for (factory, work) in [
                (Factory::Rescale(18), rescale_work),
                (Factory::GuardedSilu, activation_work),
            ] {
                let native = |width| match factory {
                    Factory::Rescale(shift) => old_rescale(width, input, shift),
                    Factory::GuardedSilu => tensor::Program::activation_guarded(
                        tensor::Activation::Silu,
                        width,
                        input,
                    )
                    .unwrap(),
                };
                let full = native(width);
                let mut expected = full
                    .work_budget()
                    .checked_mul((lanes / width) as u64)
                    .unwrap();
                if remainder != 0 {
                    expected = expected
                        .checked_add(native(remainder).work_budget())
                        .unwrap();
                }
                assert_eq!(work, expected);
                let gate = match factory {
                    Factory::Rescale(_) => None,
                    Factory::GuardedSilu => Some(ActivationWorkGate {
                        prior: WorkBudget::default(),
                    }),
                };
                let map = if let Some(gate) = gate {
                    PreparedTensorMap::prepare(lanes, input, factory, 0, |_| Ok(()), gate)
                } else {
                    PreparedTensorMap::prepare(lanes, input, factory, 0, |_| Ok(()), NoWorkGate)
                }
                .unwrap();
                assert_eq!(map.work_budget().unwrap(), work);
                assert_eq!(map.full.digest(), full.digest());
                if let Some(tail) = &map.tail {
                    assert_eq!(tail.digest(), native(remainder).digest());
                }
            }
        }
    }

    fn quoted_prepare<F: FnMut(usize) -> Result<(), Error>, G: WorkGate>(
        lanes: usize,
        input: Interval,
        factory: Factory,
        baseline: usize,
        admit: F,
        gate: G,
    ) -> (MapInventory, PreparedTensorMap) {
        let before = native_attempts();
        let inventory = construction_inventory::<F, G>(lanes, factory, baseline).unwrap();
        assert_eq!(native_attempts(), before);
        assert_eq!(inventory.full_quote, factory.quote(inventory.width).unwrap());
        assert_eq!(
            inventory.tail_quote,
            if inventory.remainder == 0 {
                None
            } else {
                Some(factory.quote(inventory.remainder).unwrap())
            }
        );
        let prepared =
            PreparedTensorMap::prepare(lanes, input, factory, baseline, admit, gate).unwrap();
        assert_eq!(
            native_attempts() - before,
            1 + usize::from(inventory.remainder != 0)
        );
        (inventory, prepared)
    }

    #[test]
    fn construction_inventory_is_shared_with_prepare_and_preserves_native_quotes() {
        let input = Interval::new(-ONE, ONE).unwrap();
        for lanes in [1, 128, 129, 256, 257] {
            let mut first = 0;
            let mut peak = 0;
            let mut calls = 0;
            let (inventory, prepared) = quoted_prepare(
                lanes,
                input,
                Factory::Rescale(18),
                73,
                |bytes| {
                    if calls == 0 {
                        first = bytes;
                    }
                    calls += 1;
                    peak = peak.max(bytes);
                    Ok(())
                },
                NoWorkGate,
            );
            assert_eq!(first, inventory.peak);
            assert_eq!(peak, first);
            assert_eq!(calls, 2 + usize::from(inventory.remainder != 0));
            assert_eq!(prepared.full.lanes(), inventory.width);
            assert_eq!(prepared.tail.is_some(), inventory.tail_quote.is_some());
            let mut first = 0;
            let mut peak = 0;
            let mut calls = 0;
            let (inventory, _) = quoted_prepare(
                lanes,
                input,
                Factory::GuardedSilu,
                73,
                |bytes| {
                    if calls == 0 {
                        first = bytes;
                    }
                    calls += 1;
                    peak = peak.max(bytes);
                    Ok(())
                },
                ActivationWorkGate {
                    prior: WorkBudget::local(),
                },
            );
            assert_eq!(first, inventory.peak);
            assert_eq!(peak, first);
            assert_eq!(calls, 2 + usize::from(inventory.remainder != 0));
        }
    }

    #[test]
    fn local_queries_reject_geometry_domain_shift_and_workspace_mismatch_before_factories() {
        let good = Interval::new(-ONE, ONE).unwrap();
        let mut rescale = tensor::SoftmaxWorkWorkspace::empty();
        let mut activation = tensor::ActivationWorkWorkspace::empty();
        let before = native_attempts();
        for lanes in [0, MAX_VALUES + 1, usize::MAX] {
            assert_eq!(
                scheduled_work(
                    Factory::Rescale(18), lanes, good,
                    MapWorkWorkspace::Rescale(&mut rescale),
                ),
                Err(Error::Bounds)
            );
            assert_eq!(
                construction_inventory::<fn(usize) -> Result<(), Error>, NoWorkGate>(
                    lanes, Factory::Rescale(18), 0,
                ).err(),
                Some(Error::Bounds)
            );
        }
        assert_eq!(
            construction_inventory::<fn(usize) -> Result<(), Error>, NoWorkGate>(
                129, Factory::Rescale(18), usize::MAX,
            ).err(),
            Some(Error::Bounds)
        );
        for shift in [0, 63, u8::MAX] {
            assert_eq!(
                scheduled_work(
                    Factory::Rescale(shift), 129, good,
                    MapWorkWorkspace::Rescale(&mut rescale),
                ),
                Err(Error::Bounds)
            );
        }
        assert_eq!(
            scheduled_work(
                Factory::GuardedSilu, 129, good,
                MapWorkWorkspace::Rescale(&mut rescale),
            ),
            Err(Error::Shape)
        );
        assert_eq!(
            scheduled_work(
                Factory::Rescale(18), 129, good,
                MapWorkWorkspace::GuardedSilu(&mut activation),
            ),
            Err(Error::Shape)
        );
        let extreme = Interval::new(i64::MIN, i64::MAX).unwrap();
        assert_eq!(
            scheduled_work(
                Factory::GuardedSilu, 129, extreme,
                MapWorkWorkspace::GuardedSilu(&mut activation),
            ),
            Err(Error::Bounds)
        );
        assert_eq!(
            scheduled_work(
                Factory::GuardedSilu, 129, good,
                MapWorkWorkspace::GuardedSilu(&mut activation),
            ),
            Err(Error::Phase)
        );
        activation.reset();
        assert!(scheduled_work(
            Factory::GuardedSilu, 129, good,
            MapWorkWorkspace::GuardedSilu(&mut activation),
        ).is_ok());
        assert_eq!(
            scheduled_work(
                Factory::Rescale(18), 129, extreme,
                MapWorkWorkspace::Rescale(&mut rescale),
            ),
            Err(Error::Bounds)
        );
        assert!(scheduled_work(
            Factory::Rescale(18), 129, good,
            MapWorkWorkspace::Rescale(&mut rescale),
        ).is_ok());
        assert_eq!(native_attempts(), before);
    }
}
