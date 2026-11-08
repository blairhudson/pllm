//! Private native RMS custody. Inventory is a PROPOSAL, not funded authority.
use super::attention_work::SoftmaxSink;
use super::rms_work::RmsSink;
use super::*;

pub(super) struct RmsArgs<'a> {
    pub(super) rows: usize,
    pub(super) width: usize,
    pub(super) input: Interval,
    pub(super) gamma: &'a [i64],
    pub(super) input_norm: Option<u128>,
}

// Separate from ResidentBuilder.state: gamma and finish require its Ready state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State { Empty, Entering, Failed, Complete }

struct Entered<'a>(&'a mut State);
impl Drop for Entered<'_> {
    fn drop(&mut self) {
        if *self.0 == State::Entering {
            *self.0 = State::Failed;
        }
    }
}

pub(super) struct RmsResidentSlot {
    state: State,
    builder: Option<ResidentBuilder>,
    program: Option<Program>,
    recipe: Option<(Value, u128)>,
}

/// Named, non-additive inventory components; deliberately no envelope total.
/// Legacy peak/Arc estimates are NOT adopted. Caller capacities, ABI frames,
/// full/tail overlap and rejected reserve capacity remain unresolved (census).
pub(super) struct RmsResidentQuote {
    pub(super) legacy: ConstructionQuote,
    pub(super) slot_bytes: usize,
    pub(super) schedule_bytes: usize,
    pub(super) row_program_bytes: usize,
    pub(super) row_builder_bytes: usize,
    pub(super) row_result_bytes: usize,
    pub(super) resident_argument_bytes: usize,
    pub(super) args_bytes: usize,
    pub(super) sink_bytes: usize,
    pub(super) builder_option_bytes: usize,
    pub(super) recipe_option_bytes: usize,
    pub(super) recipe_result_bytes: usize,
    pub(super) program_option_bytes: usize,
    pub(super) output_option_bytes: usize,
    pub(super) output_result_bytes: usize,
    pub(super) unit_result_bytes: usize,
}

impl RmsResidentSlot {
    pub(super) const fn empty() -> Self {
        Self { state: State::Empty, builder: None, program: None, recipe: None }
    }

    // Read-only proposal, never called as a new admitted-dispatch rejection.
    pub(super) fn quote(args: &RmsArgs<'_>) -> Result<RmsResidentQuote, ConstructionQuoteError> {
        rms_work::geometry(args.rows, args.width, args.gamma.len())
            .map_err(ConstructionQuoteError::Geometry)?;
        let construction = match args.input_norm {
            None => Construction::RmsNormCertified {
                rows: args.rows, width: args.width, gamma_len: args.gamma.len(),
            },
            Some(_) => Construction::RmsNormWithInputNorm {
                rows: args.rows, width: args.width, gamma_len: args.gamma.len(),
            },
        };
        Ok(RmsResidentQuote {
            legacy: construction.quote()?,
            slot_bytes: std::mem::size_of::<Self>(),
            schedule_bytes: std::mem::size_of::<ResidentNativeSchedule<'_>>(),
            row_program_bytes: std::mem::size_of::<Program>(),
            row_builder_bytes: std::mem::size_of::<Builder>(),
            row_result_bytes: std::mem::size_of::<Result<Builder, Error>>(),
            resident_argument_bytes: std::mem::size_of::<Builder>(),
            args_bytes: std::mem::size_of::<RmsArgs<'_>>(),
            sink_bytes: std::mem::size_of::<ResidentRecipeSink<'_>>(),
            builder_option_bytes: std::mem::size_of::<Option<ResidentBuilder>>(),
            recipe_option_bytes: std::mem::size_of::<Option<(Value, u128)>>(),
            recipe_result_bytes: std::mem::size_of::<Result<(Value, u128), Error>>(),
            program_option_bytes: std::mem::size_of::<Option<Program>>(),
            output_option_bytes: std::mem::size_of::<Option<FactoryOutput>>(),
            output_result_bytes: std::mem::size_of::<Result<FactoryOutput, Error>>(),
            unit_result_bytes: std::mem::size_of::<Result<(), Error>>(),
        })
    }

    pub(super) fn build_into(
        &mut self,
        args: &RmsArgs<'_>,
        out: &mut Option<FactoryOutput>,
    ) -> Result<(), Error> {
        if self.state != State::Empty {
            return Err(Error::Phase);
        }
        let Self { state, builder, program, recipe } = self;
        *state = State::Entering;
        let entered = Entered(state);
        if out.is_some() {
            return Err(Error::Binding);
        }
        #[cfg(test)]
        rms_work::entered(0);
        let lanes = rms_work::geometry(args.rows, args.width, args.gamma.len())?;
        // row returns only empty Vec headers. No hook/fallibility after return
        // and before installation; acquisition of backing starts later in gamma.
        *builder = Some(ResidentBuilder::from_builder(Builder::row(lanes, args.input)?));
        let resident = builder.as_mut().ok_or(Error::Phase)?;
        #[cfg(test)]
        hook(RmsStage::Installed)?;
        resident.builder.as_mut().ok_or(Error::Phase)?.0.numeric_profile = RMS_NORM_PROFILE;
        #[cfg(test)]
        hook(RmsStage::Recipe)?;
        *recipe = Some(numeric::rms_recipe(
            &mut ResidentRecipeSink(resident), args.width, args.input,
            args.gamma, true, args.input_norm,
        )?);
        #[cfg(test)]
        hook(RmsStage::PostRecipe)?;
        let (output, squared) = recipe.as_ref().copied().ok_or(Error::Phase)?;
        resident.finish(output)?;
        resident.transfer_complete(program)?;
        #[cfg(test)]
        hook(RmsStage::PostTransfer)?;
        work_preflight::commit_rms_output(program, squared, out)?;
        *entered.0 = State::Complete;
        Ok(())
    }
}

// Shared native recipe lens. It borrows the installed owner, never transfers it
// into a recipe-local Builder. RMS and rotary use the same primitive machinery.
pub(super) struct ResidentRecipeSink<'a>(pub(super) &'a mut ResidentBuilder);

impl ResidentRecipeSink<'_> {
    // Recipe never transfers/removes this Builder. Only finish does so, after
    // this borrow ends. These accesses have the same Builder invariants as before.
    fn builder(&self) -> &Builder {
        self.0.builder.as_ref().expect("installed RMS builder")
    }
    fn builder_mut(&mut self) -> &mut Builder {
        self.0.builder.as_mut().expect("installed RMS builder")
    }
}

impl SoftmaxSink for ResidentRecipeSink<'_> {
    fn node(&self, v: Value, bit: bool) -> Result<Node, Error> {
        SoftmaxSink::node(self.builder(), v, bit)
    }
    fn push(&mut self, i: Instruction, b: Interval, bit: bool) -> Result<Value, Error> {
        SoftmaxSink::push(self.builder_mut(), i, b, bit)
    }
    fn refine(&mut self, v: Value, b: Interval) {
        SoftmaxSink::refine(self.builder_mut(), v, b)
    }
    fn compact(&self) -> bool { SoftmaxSink::compact(self.builder()) }
    fn sign_compact(&mut self, v: Value) -> Result<Value, Error> {
        SoftmaxSink::sign_compact(self.builder_mut(), v)
    }
    fn input(&self) -> Value { SoftmaxSink::input(self.builder()) }
    fn width(&self, v: Value) -> Result<u8, Error> { SoftmaxSink::width(self.builder(), v) }
    fn affine(&mut self, a: Value, b: Option<Value>, c: Affine) -> Result<Value, Error> {
        SoftmaxSink::affine(self.builder_mut(), a, b, c)
    }
    fn product(&mut self, a: Value, b: Value, rescale: bool) -> Result<Value, Error> {
        SoftmaxSink::product(self.builder_mut(), a, b, rescale)
    }
    fn square(&mut self, v: Value) -> Result<Value, Error> {
        SoftmaxSink::square(self.builder_mut(), v)
    }
    fn rescale(&mut self, v: Value, shift: u8) -> Result<Value, Error> {
        SoftmaxSink::rescale(self.builder_mut(), v, shift)
    }
    fn row_map(&mut self, v: Value, width: usize, map: RowMap) -> Result<Value, Error> {
        SoftmaxSink::row_map(self.builder_mut(), v, width, map)
    }
    fn sign(&mut self, v: Value) -> Result<Value, Error> {
        SoftmaxSink::sign(self.builder_mut(), v)
    }
    fn bits(&mut self, v: Value) -> Result<Value, Error> {
        SoftmaxSink::bits(self.builder_mut(), v)
    }
    fn bit_map(&mut self, v: Value, map: BitMap) -> Result<Value, Error> {
        SoftmaxSink::bit_map(self.builder_mut(), v, map)
    }
    fn bit_or(&mut self, a: Value, b: Value) -> Result<Value, Error> {
        SoftmaxSink::bit_or(self.builder_mut(), a, b)
    }
    fn bit_xor(&mut self, a: Value, b: Value) -> Result<Value, Error> {
        SoftmaxSink::bit_xor(self.builder_mut(), a, b)
    }
    fn from_bit(&mut self, v: Value) -> Result<Value, Error> {
        SoftmaxSink::from_bit(self.builder_mut(), v)
    }
    fn from_boolean(&mut self, v: Value) -> Result<Value, Error> {
        SoftmaxSink::from_boolean(self.builder_mut(), v)
    }
    fn highest_bit(&mut self, v: Value, width: u8) -> Result<Value, Error> {
        SoftmaxSink::highest_bit(self.builder_mut(), v, width)
    }
}

impl RmsSink for ResidentRecipeSink<'_> {
    fn gamma(&mut self, input: Value, width: usize, gamma: &[i64]) -> Result<Value, Error> {
        let scales: [Affine; MAX_ROW_LANES] = std::array::from_fn(|i| Affine {
            left: gamma[i % width], right: 0, constant: 0,
        });
        let lanes = self.node(input, false)?.lanes;
        #[cfg(test)]
        observe_gamma(gamma);
        let output = self.0.affine_lanes(input, &scales[..lanes])?;
        #[cfg(test)]
        hook(RmsStage::PostAffine)?;
        Ok(output)
    }
    fn checked_profile(&mut self) { RmsSink::checked_profile(self.builder_mut()) }
    fn constant_xor(&mut self, v: Value, c: u64) -> Result<Value, Error> {
        RmsSink::constant_xor(self.builder_mut(), v, c)
    }
    fn outside_bound(&mut self, v: Value, bound: i64, below: bool) -> Result<Value, Error> {
        RmsSink::outside_bound(self.builder_mut(), v, bound, below)
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RmsStage { Installed, Recipe, PostAffine, PostRecipe, PostTransfer }

#[cfg(test)]
thread_local! {
    static FAULT: std::cell::Cell<Option<(RmsStage, usize, bool)>> = const { std::cell::Cell::new(None) };
    static GAMMA: std::cell::Cell<(*const i64, usize)> = const { std::cell::Cell::new((std::ptr::null(), 0)) };
    // Track a real original table, not a sidecar lifetime sentinel.
    static BACKING: std::cell::Cell<(*const Affine, usize, usize)> = const { std::cell::Cell::new((std::ptr::null(), 0, 0)) };
    static OBSERVE_NEXT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static SECOND_BACKING: std::cell::Cell<(*const Affine, usize, usize)> = const { std::cell::Cell::new((std::ptr::null(), 0, 0)) };
    static OBSERVE_SECOND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn hook(stage: RmsStage) -> Result<(), Error> {
    FAULT.with(|fault| {
        if let Some((wanted, remaining, panic)) = fault.get() {
            if wanted == stage {
                if remaining == 1 {
                    fault.set(None);
                    if panic { panic!("test resident RMS stage unwind"); }
                    return Err(Error::Bounds);
                }
                fault.set(Some((wanted, remaining - 1, panic)));
            }
        }
        Ok(())
    })
}

#[cfg(test)]
fn observe_gamma(gamma: &[i64]) { GAMMA.with(|g| g.set((gamma.as_ptr(), gamma.len()))); }

#[cfg(test)]
pub(super) fn observe_affine(table: &[Affine]) {
    if OBSERVE_NEXT.with(|next| next.replace(false)) {
        BACKING.with(|backing| backing.set((table.as_ptr(), table.len(), 0)));
    } else if OBSERVE_SECOND.with(|next| next.replace(false)) {
        SECOND_BACKING.with(|backing| backing.set((table.as_ptr(), table.len(), 0)));
    }
}

// Test-only destruction plumbing: destroy the actual exact owner first, then record
// completion for that pointer. No production Drop change, allocator/RSS claim,
// Arc-alias coverage or guarantee against abort/double panic is implied.
#[cfg(test)]
fn teardown<T>(pointer: *const Affine, capacity: usize, owned: T) {
    drop(owned);
    BACKING.with(|backing| {
        let (watched, watched_capacity, drops) = backing.get();
        if pointer == watched && capacity == watched_capacity && !watched.is_null() {
            backing.set((watched, watched_capacity, drops + 1));
        }
    });
    SECOND_BACKING.with(|backing| {
        let (watched, watched_capacity, drops) = backing.get();
        if pointer == watched && capacity == watched_capacity && !watched.is_null() {
            backing.set((watched, watched_capacity, drops + 1));
        }
    });
}

#[cfg(test)]
impl Drop for Program {
    fn drop(&mut self) {
        for table in &mut self.public_affines {
            teardown(table.as_ptr(), table.len(), std::mem::take(table));
        }
    }
}

#[cfg(test)]
impl Drop for ResidentBuilder {
    fn drop(&mut self) {
        teardown(self.pending_affine.as_ptr(), self.pending_affine.extent(),
            std::mem::replace(&mut self.pending_affine, ExactStorage::empty()));
    }
}

#[cfg(test)]
#[path = "tensor_rms_resident_tests.rs"]
mod tests;
