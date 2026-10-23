//! Work preflight for the original eager/guarded activation recipe. No Program,
//! native factory, private input, entropy or execution owner is constructed.
use super::attention_work::SoftmaxSink;
use super::*;

#[cfg(test)]
thread_local! {
    static ENTRIES: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
}
#[cfg(test)]
pub(super) fn entered(kind: Activation, guarded: bool) {
    ENTRIES.with(|entries| {
        let mut values = entries.get();
        let index = usize::from(kind == Activation::Silu) * 2 + usize::from(guarded);
        values[index] += 1;
        entries.set(values);
    });
}
#[cfg(test)]
pub(super) fn entries() -> [usize; 4] {
    ENTRIES.with(std::cell::Cell::get)
}

/// Caller-owned fixed native metadata. Quote `workspace_bytes()` BEFORE
/// `empty()`. No Program, coefficient tables, growing collection or input owner.
/// Native public-proof CPU/callframes, construction/execution buffers and caller
/// wrapper owners remain separately priced, not covered by this inline quote.
/// Error poisons until explicit reset. Reset never refunds a caller work ledger.
pub struct ActivationWorkWorkspace {
    metadata: SoftmaxWorkWorkspace,
    poisoned: bool,
}

impl ActivationWorkWorkspace {
    pub const fn workspace_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
    pub const fn empty() -> Self {
        Self {
            metadata: SoftmaxWorkWorkspace::empty(),
            poisoned: false,
        }
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    pub fn reset(&mut self) {
        self.metadata.clear();
        self.poisoned = false;
    }
    #[cfg(test)]
    pub(super) fn assert_native_metadata(&self, program: &Program) {
        self.metadata.assert_native_metadata(program);
    }
    fn query(
        &mut self,
        kind: Activation,
        lanes: usize,
        input: Interval,
        guarded: bool,
    ) -> Result<WorkBudget, Error> {
        if self.poisoned {
            return Err(Error::Phase);
        }
        let result = (|| {
            // Exactly Builder::new's geometry validation, before recipe/domain
            // validation. There is no eager construction just to discover work.
            validate_tile_lanes(lanes)?;
            self.metadata.reset(lanes, input);
            let output = emit(&mut self.metadata, kind, input, guarded, false)?;
            self.metadata.finish_work(output)
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}

impl Program {
    /// Original eager SiLU/GELU WorkBudget, not a compact/profile substitute.
    /// Same primitive validation and cost owner as the actual native Builder.
    /// No Budget, source/range proof, factory permit or request authority.
    /// Caller checked-multiplies actual full invocations, checked-adds the tail
    /// ONCE and all prior/other work, then admits against real numeric ceilings
    /// before any factory. A reset cannot erase that independent aggregate.
    pub fn activation_work(
        kind: Activation,
        lanes: usize,
        input: Interval,
        workspace: &mut ActivationWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        workspace.query(kind, lanes, input, false)
    }
    /// Original guarded activation, including its early secret predicates,
    /// exact safe-domain guard, unchanged signed truncations and output proof.
    /// WorkBudget's private/AES fields remain per-role envelopes, not a timing
    /// estimate or the sum of independently maximized per-instruction roles.
    pub fn activation_guarded_work(
        kind: Activation,
        lanes: usize,
        input: Interval,
        workspace: &mut ActivationWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        workspace.query(kind, lanes, input, true)
    }
}

// The original recipe, shared ONCE by native Builder and metadata preflight.
// The existing compact caller keeps its original flag/profile; public queries
// above deliberately expose only the selected full-ring legacy implementations.
pub(super) fn emit<S: SoftmaxSink>(
    b: &mut S,
    kind: Activation,
    input: Interval,
    guarded: bool,
    compact: bool,
) -> Result<Value, Error> {
    let original = b.input();
    let pivots = match kind {
        Activation::Gelu => [-1048576, -511180, 786432],
        Activation::Silu => [-2097152, -1048576, 1048576],
    };
    let early = if guarded {
        Some(activation_predicates(b, original, pivots, compact)?)
    } else {
        None
    };
    let x = if let Some(predicates) = early {
        let inside = b.affine(
            predicates[2],
            Some(predicates[0]),
            Affine {
                left: 1,
                right: -1,
                constant: 0,
            },
        )?;
        // [x<=hi]-[x<lo] is exactly the indicator of lo<=x<=hi.
        b.refine(inside, Interval::new(0, 1)?);
        let safe = b.product(original, inside, false)?;
        // Excluded tails become zero only inside the polynomial. The final
        // identity branch still consumes the untouched original input.
        let lo = input.lower().max(pivots[0]);
        let hi = input.upper().min(pivots[2]);
        b.refine(
            safe,
            if lo <= hi {
                Interval::new(lo.min(0), hi.max(0))?
            } else {
                Interval::new(0, 0)?
            },
        );
        safe
    } else {
        original
    };
    let x2 = b.product(x, x, true)?;
    let x3 = if kind == Activation::Gelu {
        Some(b.product(x2, x, true)?)
    } else {
        None
    };
    let x4 = b.product(x2, x2, true)?;
    let x6 = match x3 {
        Some(v) => b.product(v, v, true)?,
        None => b.product(x2, x4, true)?,
    };
    let (low, high, pivots) = match kind {
        Activation::Gelu => (
            polynomial(
                b,
                x,
                -132488,
                &[(x, -110694), (x2, -30952), (x3.unwrap(), -2892)],
            )?,
            polynomial(
                b,
                x,
                2235,
                &[(x, 131072), (x2, 94458), (x4, -9879), (x6, 473)],
            )?,
            [-1048576, -511180, 786432],
        ),
        Activation::Silu => (
            polynomial(b, x, -80413, &[(x, -21489), (x2, -1453)])?,
            polynomial(
                b,
                x,
                2229,
                &[(x, 131072), (x2, 59806), (x4, -2913), (x6, 71)],
            )?,
            [-2097152, -1048576, 1048576],
        ),
    };
    let predicates = match early {
        Some(predicates) => predicates,
        None => activation_predicates(b, original, pivots, compact)?,
    };
    let upper = select(b, predicates[2], high, original)?;
    let middle = select(b, predicates[1], low, upper)?;
    let removed = b.product(predicates[0], middle, false)?;
    let output = b.affine(
        middle,
        Some(removed),
        Affine {
            left: 1,
            right: -1,
            constant: 0,
        },
    )?;
    if guarded {
        b.refine(output, numeric::activation_bounds(kind, input)?);
    }
    Ok(output)
}
