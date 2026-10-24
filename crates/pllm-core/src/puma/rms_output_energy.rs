//! Conditional energy theorem for the ORIGINAL epsilon-one Q18 RMS recipe.
//! No GENERAL/input certificate, request admission, or attention conversion.
use super::*;
use std::mem::size_of;
use std::sync::{atomic::AtomicUsize, Arc};

const Q: u128 = 1 << 18;
const EXPONENTS: usize = 35;

fn add(a: u128, b: u128) -> Result<u128, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn mul(a: u128, b: u128) -> Result<u128, Error> {
    a.checked_mul(b).ok_or(Error::Bounds)
}
fn root(a: u128) -> u128 {
    let r = a.isqrt();
    r + u128::from(r * r != a)
}
fn magnitude(a: Interval) -> u128 {
    u128::from(a.lower().unsigned_abs().max(a.upper().unsigned_abs()))
}

/// Immutable native arguments. Energy is a CONDITIONAL complete-row premise;
/// callers must obtain actual GENERAL authority independently.
pub struct RmsEnergyParameters<'a> {
    rows: usize,
    width: usize,
    input: Interval,
    gamma: &'a [i64],
    energy: u128,
}

/// Named logical storage/work only, not compiled stack, allocator/RSS or CPU.
/// Original gamma backing/owner and enclosing caller owners remain additional.
#[derive(Clone, Copy, Debug)]
pub struct RmsEnergyResources {
    pub known_bytes: usize,
    /// Conservative sum of theorem controls and existing native construction
    /// phase. Caller originals, opaque allocation/stack/RSS remain additional.
    pub known_new_peak_bytes: usize,
    pub denominator_cases: usize,
    pub polynomial_cells: usize,
    pub gamma_visits: usize,
    pub metadata_node_visits: usize,
    pub native_construction: ConstructionQuote,
}
#[derive(Clone, Copy, Debug)]
pub struct RmsEnergyLimits {
    pub known_bytes: usize,
    pub denominator_cases: usize,
    pub polynomial_cells: usize,
}
impl RmsEnergyResources {
    pub fn limits(self) -> RmsEnergyLimits {
        RmsEnergyLimits {
            known_bytes: self.known_bytes,
            denominator_cases: self.denominator_cases,
            polynomial_cells: self.polynomial_cells,
        }
    }
    fn check(self, limits: RmsEnergyLimits) -> Result<(), Error> {
        if self.known_bytes > limits.known_bytes
            || self.denominator_cases > limits.denominator_cases
            || self.polynomial_cells > limits.polynomial_cells
        {
            return Err(Error::Bounds);
        }
        Ok(())
    }
}

pub struct RmsEnergyWorkspace {
    validation: RmsWorkWorkspace,
    inverse: SoftmaxWorkWorkspace,
    limits: RmsEnergyLimits,
    poisoned: bool,
}
impl RmsEnergyWorkspace {
    /// Quote/check BEFORE fixed metadata construction. No private data/factory.
    pub fn new(p: &RmsEnergyParameters<'_>, limits: RmsEnergyLimits) -> Result<Self, Error> {
        p.resources()?.check(limits)?;
        Ok(Self {
            validation: RmsWorkWorkspace::empty(),
            inverse: SoftmaxWorkWorkspace::empty(),
            limits,
            poisoned: false,
        })
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Does not refund external work or request/native state.
    pub fn reset(&mut self) {
        self.validation.reset();
        self.inverse.clear();
        self.poisoned = false;
    }
}

impl<'a> RmsEnergyParameters<'a> {
    pub fn new(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &'a [i64],
        energy: u128,
    ) -> Result<Self, Error> {
        rms_work::geometry(rows, width, gamma.len())?;
        let p = Self {
            rows,
            width,
            input,
            gamma,
            energy,
        };
        p.domain()?;
        Ok(p)
    }
    fn interval_energy(&self) -> Result<u128, Error> {
        mul(
            mul(magnitude(self.input), magnitude(self.input))?,
            self.width as u128,
        )
    }
    fn domain(&self) -> Result<Interval, Error> {
        let minimum = if self.input.lower() <= 0 && self.input.upper() >= 0 {
            0
        } else {
            u128::from(
                self.input
                    .lower()
                    .unsigned_abs()
                    .min(self.input.upper().unsigned_abs()),
            )
        };
        let minimum_square = mul(minimum, minimum)?;
        if self.energy < mul(minimum_square, self.width as u128)? {
            return Err(Error::Bounds);
        }
        let energy = self.energy.min(self.interval_energy()?);
        let sum_lo = mul(minimum_square / Q, self.width as u128)?;
        let sum_hi = add(energy, mul(self.width as u128, Q - 1)?)? / Q;
        let c = numeric::rms_mean_coefficient(self.width)? as u128;
        let epsilon = RMS_NORM_EPSILON_Q18 as u128;
        let lo = add(mul(sum_lo, c)? / Q, epsilon)?;
        let hi = add(mul(sum_hi, c)?.div_ceil(Q), epsilon)?;
        if lo == 0 || hi >= 1_u128 << EXPONENTS || lo > hi {
            return Err(Error::Bounds);
        }
        Interval::new(
            i64::try_from(lo).map_err(|_| Error::Bounds)?,
            i64::try_from(hi).map_err(|_| Error::Bounds)?,
        )
    }
    pub fn resources(&self) -> Result<RmsEnergyResources, Error> {
        let domain = self.domain()?;
        let mut cases = 0usize;
        for j in 0..EXPONENTS {
            let lo = domain.lower().max(1_i64 << j);
            let hi = domain.upper().min((1_i64 << (j + 1)) - 1);
            if lo <= hi {
                let n = (hi - lo + 1) as u64;
                cases += n.div_ceil(n.div_ceil(8)) as usize;
            }
        }
        let native = Construction::RmsNormWithInputNorm {
            rows: self.rows,
            width: self.width,
            gamma_len: self.gamma.len(),
        }
        .quote()
        .map_err(|_| Error::Bounds)?;
        // Explicit conservative UNION of logical phase slots, including returned
        // and caller bindings, actual Result/argument copies and native proof
        // scratch. No padding is used to stand in for private callframes.
        let slots = [
            2 * size_of::<RmsEnergyWorkspace>(),
            size_of::<Result<RmsEnergyWorkspace, Error>>(),
            size_of::<RmsWorkWorkspace>(),
            size_of::<SoftmaxWorkWorkspace>(),
            3 * size_of::<Self>(),
            2 * size_of::<Result<Self, Error>>(),
            3 * size_of::<RmsEnergyResources>(),
            size_of::<Result<RmsEnergyResources, Error>>(),
            3 * size_of::<RmsEnergyLimits>(),
            2 * size_of::<ConditionalRmsEnergy<'a>>(),
            size_of::<Result<ConditionalRmsEnergy<'a>, Error>>(),
            2 * size_of::<PreparedRmsEnergy<'a>>(),
            size_of::<Result<PreparedRmsEnergy<'a>, Error>>(),
            2 * size_of::<OriginalRmsEnergyOwner<'a>>(),
            size_of::<Result<OriginalRmsEnergyOwner<'a>, Error>>(),
            2 * size_of::<FactoryOutput>(),
            size_of::<Result<FactoryOutput, Error>>(),
            2 * size_of::<Arc<Program>>(),
            2 * size_of::<AtomicUsize>(),
            // Moving Program into Arc retains original return/destination slots.
            size_of::<Program>(),
            native.proof_buffer_bytes,
            12 * size_of::<u128>(),
            10 * size_of::<usize>(),
            6 * size_of::<Interval>(),
            4 * size_of::<Result<Interval, Error>>(),
            4 * size_of::<Result<u128, Error>>(),
            size_of::<std::slice::Iter<'a, i64>>(),
        ];
        let known_bytes = slots
            .into_iter()
            .try_fold(0usize, |a, b| a.checked_add(b).ok_or(Error::Bounds))?;
        Ok(RmsEnergyResources {
            known_bytes,
            known_new_peak_bytes: known_bytes
                .checked_add(native.new_peak_bytes)
                .ok_or(Error::Bounds)?,
            denominator_cases: cases,
            polynomial_cells: (cases + EXPONENTS) * (((1 << 17) - (1 << 16)) / 256 + 1),
            gamma_visits: self
                .rows
                .checked_mul(self.width)
                .and_then(|v| self.width.checked_mul(3)?.checked_add(v))
                .ok_or(Error::Bounds)?,
            metadata_node_visits: (cases + 1)
                .checked_mul(MAX_NODES)
                .and_then(|n| n.checked_mul(36))
                .ok_or(Error::Bounds)?,
            native_construction: native,
        })
    }
    pub fn prove(
        self,
        workspace: &mut RmsEnergyWorkspace,
    ) -> Result<ConditionalRmsEnergy<'a>, Error> {
        if workspace.poisoned {
            return Err(Error::Phase);
        }
        let result = (|| {
            self.resources()?.check(workspace.limits)?;
            // ORIGINAL validation/cost traversal, not Program construction.
            Program::rms_norm_with_input_norm_work(
                self.rows,
                self.width,
                self.input,
                self.gamma,
                self.energy,
                &mut workspace.validation,
            )?;
            let domain = self.domain()?;
            let c = numeric::rms_mean_coefficient(self.width)? as u128;
            let mut gamma = 0_u128;
            let mut nonzero = 0_u128;
            for &coefficient in self.gamma {
                gamma = gamma.max(u128::from(coefficient.unsigned_abs()));
                nonzero += u128::from(coefficient != 0);
            }
            let mut squared = 0;
            for j in 0..EXPONENTS {
                let lo = domain.lower().max(1_i64 << j);
                let hi = domain.upper().min((1_i64 << (j + 1)) - 1);
                if lo > hi {
                    continue;
                }
                let stride = ((hi - lo + 1) as u64).div_ceil(8) as i64;
                let mut start = lo;
                while start <= hi {
                    let end = hi.min(start + stride - 1);
                    let inverse = numeric::rms_energy_inverse_bounds(
                        Interval::new(start, end)?,
                        &mut workspace.inverse,
                    )?;
                    // D<=end => S*c < (end-epsilon+1)*Q. Same-row correlation;
                    // every stochastic truncation lies between floor and ceil.
                    let sum_max = (mul((end - RMS_NORM_EPSILON_Q18 + 1) as u128, Q)? - 1) / c;
                    let energy = self
                        .energy
                        .min(self.interval_energy()?)
                        .min(add(mul(sum_max, Q)?, mul(self.width as u128, Q - 1)?)?);
                    let output = if energy == 0 || gamma == 0 {
                        0
                    } else {
                        let r = magnitude(inverse);
                        let norm = add(
                            root(mul(mul(energy, r)?, r)?).div_ceil(Q),
                            root(self.width as u128),
                        )?;
                        let radius = add(mul(norm, gamma)?.div_ceil(Q), root(nonzero))?;
                        mul(radius, radius)?
                    };
                    squared = squared.max(output);
                    start = end + 1;
                }
            }
            Ok(ConditionalRmsEnergy {
                parameters: self,
                squared,
            })
        })();
        if result.is_err() {
            workspace.poisoned = true;
        }
        result
    }
}

pub struct ConditionalRmsEnergy<'a> {
    parameters: RmsEnergyParameters<'a>,
    squared: u128,
}
impl ConditionalRmsEnergy<'_> {
    pub fn conditional_squared_upper(&self) -> u128 {
        self.squared
    }
}

/// Captured admitted ticket; no replacement gamma/energy/program arguments.
pub struct PreparedRmsEnergy<'a> {
    admitted: AdmittedFactory<'a>,
    theorem: ConditionalRmsEnergy<'a>,
}
pub(super) fn prepare<'a>(
    admitted: AdmittedFactory<'a>,
    parameters: RmsEnergyParameters<'a>,
    workspace: &mut RmsEnergyWorkspace,
) -> Result<PreparedRmsEnergy<'a>, Error> {
    let theorem = parameters.prove(workspace)?;
    Ok(PreparedRmsEnergy { admitted, theorem })
}
impl<'a> PreparedRmsEnergy<'a> {
    /// Call only after caller admits quoted native+Arc/control transition peak.
    /// Metadata proof is prepaid separately; this consumes the numeric ticket.
    pub fn build(self) -> Result<OriginalRmsEnergyOwner<'a>, Error> {
        let expected = self.admitted.summary().unit_work;
        let output = self.admitted.build()?;
        let (program, certificate) = output.into_parts();
        if !matches!(certificate, ProducedCertificate::GeneralOutputRowSquared(_)) {
            return Err(Error::Binding);
        }
        if program.work_budget() != expected {
            return Err(Error::Binding);
        }
        Ok(OriginalRmsEnergyOwner {
            program: Arc::new(program),
            theorem: self.theorem,
            certificate,
        })
    }
}

/// Creation-issued original native Arc + original parameter borrow. NOT GENERAL
/// publication authority: actual input premise still needs compiler discharge.
/// No interner/digest rebinding or foreign-program constructor exists.
///
/// Original coefficient allocation cannot change while owner remains live:
/// ```compile_fail
/// use pllm_core::puma::{Interval,WorkBudget,multiply::tensor::*};
/// let mut gamma = [1_i64 << 18;2];
/// let input = Interval::new(0,0).unwrap();
/// let parameters = RmsEnergyParameters::new(1,2,input,&gamma,0).unwrap();
/// let mut proof = RmsEnergyWorkspace::new(&parameters,parameters.resources().unwrap().limits()).unwrap();
/// let factory = Construction::RmsNormWithInputNorm{rows:1,width:2,gamma_len:2}
///     .bind(FactoryParameters::RmsNormWithInputNorm{input,gamma:&gamma,squared_norm:0}).unwrap();
/// let mut work = WorkPreflightWorkspace::empty();
/// let ticket = factory.preflight(1,&mut work).unwrap();
/// let mut total = WorkBudget::default();
/// let (part,_) = CallerWorkLedger::new(&mut total,u64::MAX,u64::MAX).unwrap().admit(ticket).unwrap().into_parts();
/// let owner = part.prepare_rms_energy(&mut proof).unwrap().build().unwrap();
/// gamma[0] = 0;
/// let _ = owner.program();
/// ```
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::OriginalRmsEnergyOwner;
/// fn escape<'a>(owner: OriginalRmsEnergyOwner<'a>) -> OriginalRmsEnergyOwner<'static> {
///     owner
/// }
/// ```
pub struct OriginalRmsEnergyOwner<'a> {
    program: Arc<Program>,
    theorem: ConditionalRmsEnergy<'a>,
    certificate: ProducedCertificate,
}
impl OriginalRmsEnergyOwner<'_> {
    pub fn program(&self) -> &Arc<Program> {
        &self.program
    }
    pub fn original_certificate(&self) -> ProducedCertificate {
        self.certificate
    }
    pub fn conditional_squared_upper(&self) -> u128 {
        self.theorem.squared
    }
    pub fn check_native(&self, other: &Arc<Program>) -> Result<(), Error> {
        if Arc::ptr_eq(&self.program, other) {
            Ok(())
        } else {
            Err(Error::Binding)
        }
    }
    pub fn check_gamma(&self, gamma: &[i64]) -> Result<(), Error> {
        if std::ptr::eq(self.theorem.parameters.gamma, gamma) {
            Ok(())
        } else {
            Err(Error::Binding)
        }
    }
}

#[cfg(test)]
#[path = "rms_output_energy_tests.rs"]
mod tests;
