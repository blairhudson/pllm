//! Shared canonical-residue arithmetic. Callers validate modulus and operands.

pub(crate) fn mul(a: u64, b: u64, p: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(p)) as u64
}
pub(crate) fn sub(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { p - (b - a) }
}

// Private, statically dispatched observation of the same arithmetic bodies.
// Production callers supply only Noop; there are no incoming callbacks.
#[derive(Clone, Copy)]
enum RootEvent {
    Exponentiation, ExponentStep, MrSquaring, ModularProduct,
    TrialRemainder, BaseReduction, CandidateIteration, MinimumScanVisit,
    ValidationPredicate, ValidationRemainder, ProjectionDivision, MinimumScanProduct,
}

trait RootObserver {
    fn event(&mut self, event: RootEvent);
}

struct Noop;

impl RootObserver for Noop {
    fn event(&mut self, _: RootEvent) {}
}

pub(crate) fn pow(a: u64, exponent: u64, q: u64) -> u64 {
    observed_pow(a, exponent, q, &mut Noop)
}

fn observed_pow<O: RootObserver>(mut a: u64, mut exponent: u64, q: u64, o: &mut O) -> u64 {
    o.event(RootEvent::Exponentiation);
    let mut result = 1;
    while exponent != 0 {
        o.event(RootEvent::ExponentStep);
        if exponent & 1 != 0 {
            o.event(RootEvent::ModularProduct);
            result = mul(result, a, q);
        }
        o.event(RootEvent::ModularProduct);
        a = mul(a, a, q);
        exponent >>= 1;
    }
    result
}

// Deterministic Miller–Rabin for the FULL u64 domain, not just small limbs.
// Seven-base computational record: https://miller-rabin.appspot.com/ .
// The math42 record is accepted here, not an independently reproduced
// certificate. This is not a probabilistic claim or a public prime assertion API.
pub(crate) fn prime(q: u64) -> bool {
    observed_prime(q, &mut Noop)
}

fn observed_prime<O: RootObserver>(q: u64, o: &mut O) -> bool {
    if q < 2 {
        return false;
    }
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        o.event(RootEvent::TrialRemainder);
        if q % p == 0 {
            return q == p;
        }
    }
    let s = (q - 1).trailing_zeros();
    let d = (q - 1) >> s;
    for base in [2, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        o.event(RootEvent::BaseReduction);
        let a = base % q;
        if a == 0 {
            continue;
        }
        let mut x = observed_pow(a, d, q, o);
        if x == 1 || x == q - 1 {
            continue;
        }
        let mut passed = false;
        for _ in 1..s {
            o.event(RootEvent::MrSquaring);
            o.event(RootEvent::ModularProduct);
            x = mul(x, x, q);
            if x == q - 1 {
                passed = true;
                break;
            }
        }
        if !passed {
            return false;
        }
    }
    true
}

/// Public, variable-time algebra parameters; never keys or activation authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RootRequest {
    pub q: u64,
    pub m: u64,
    pub candidate_max: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RootError {
    InvalidModulus,
    InvalidOrder,
    InvalidCandidateRange,
    Composite,
    QuoteOverflow,
    Exhausted,
}

/// Descriptive constant auxiliary storage, not a physical stack-byte estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RootAuxiliaryStorage {
    Constant,
}

/// Checked upper bounds on nested logical events, including early refusals.
/// Exponent steps and MR squarings also contribute modular products; do not
/// sum these as independent prices. Validation counts are fixed upper bounds.
/// MR comparisons, loop controls, and quote arithmetic are not priced as whole
/// bodies. This is neither an exhaustive instruction/CPU total nor funding
/// authority, HE/native issuance, or a security-adoption contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RootWorkQuote {
    pub exponentiations: u64,
    pub exponent_steps: u64,
    pub mr_squarings: u64,
    pub modular_products: u64,
    pub trial_remainders: u64,
    pub base_reductions: u64,
    pub candidate_iterations: u64,
    pub minimum_scan_visits: u64,
    pub validation_predicates: u64,
    pub validation_remainders: u64,
    pub projection_divisions: u64,
    pub heap_bytes: u64,
    pub recursion_depth: u64,
    pub auxiliary_storage: RootAuxiliaryStorage,
}

fn prepare_root<O: RootObserver>(r: RootRequest, o: &mut O) -> Result<RootWorkQuote, RootError> {
    o.event(RootEvent::ValidationPredicate);
    if r.q < 3 { return Err(RootError::InvalidModulus); }
    o.event(RootEvent::ValidationPredicate);
    if r.q & 1 == 0 { return Err(RootError::InvalidModulus); }
    o.event(RootEvent::ValidationPredicate);
    if r.m < 2 { return Err(RootError::InvalidOrder); }
    o.event(RootEvent::ValidationPredicate);
    if !r.m.is_power_of_two() { return Err(RootError::InvalidOrder); }
    o.event(RootEvent::ValidationPredicate);
    o.event(RootEvent::ValidationRemainder);
    if (r.q - 1) % r.m != 0 { return Err(RootError::InvalidOrder); }
    o.event(RootEvent::ValidationPredicate);
    if r.candidate_max < 2 { return Err(RootError::InvalidCandidateRange); }
    o.event(RootEvent::ValidationPredicate);
    if r.candidate_max >= r.q { return Err(RootError::InvalidCandidateRange); }

    let overflow = RootError::QuoteOverflow;
    let c = r.candidate_max.checked_sub(1).ok_or(overflow)?;
    let h = r.m >> 1;
    let p = c.checked_mul(3).and_then(|v| v.checked_add(7)).ok_or(overflow)?;
    let steps = p.checked_mul(64).ok_or(overflow)?;
    let products = p.checked_mul(128)
        .and_then(|v| v.checked_add(441))
        .and_then(|v| v.checked_add(h)).ok_or(overflow)?;
    Ok(RootWorkQuote {
        exponentiations: p, exponent_steps: steps, mr_squarings: 441,
        modular_products: products, trial_remainders: 12, base_reductions: 7,
        candidate_iterations: c, minimum_scan_visits: h,
        validation_predicates: 7, validation_remainders: 1, projection_divisions: 1,
        heap_bytes: 0, recursion_depth: 0, auxiliary_storage: RootAuxiliaryStorage::Constant,
    })
}

/// Validate the request and finish every checked quote calculation before any
/// primality or search work. A quote does not assert that q is prime or that
/// this candidate interval contains a root-producing candidate.
pub fn public_root_work_quote(request: RootRequest) -> Result<RootWorkQuote, RootError> {
    prepare_root(request, &mut Noop)
}

/// Select the minimum primitive m-th root, using ascending candidates 2..=max.
/// After a successful projection, odd powers enumerate all primitive roots:
/// r^m=1 and r^(m/2)=-1 imply exact order m for power-of-two m in a prime
/// field. Thus the minimum agrees with OpenFHE's minimum-root convention;
/// this helper is not routed into OpenFHE, HE, or native/security admission.
/// No entropy, allocation, recursion, fallback, or retry. Variable-time PUBLIC
/// parameters only. Exhaustion is explicit even when the field contains roots.
pub fn canonical_power_two_root(request: RootRequest) -> Result<u64, RootError> {
    observed_root(request, &mut Noop)
}

fn observed_root<O: RootObserver>(r: RootRequest, o: &mut O) -> Result<u64, RootError> {
    let _quote = prepare_root(r, o)?;
    if !observed_prime(r.q, o) { return Err(RootError::Composite); }
    o.event(RootEvent::ProjectionDivision);
    let quotient = (r.q - 1) / r.m;
    for a in 2..=r.candidate_max {
        o.event(RootEvent::CandidateIteration);
        let root = observed_pow(a, quotient, r.q, o);
        if observed_pow(root, r.m, r.q, o) == 1
            && observed_pow(root, r.m >> 1, r.q, o) == r.q - 1
        {
            o.event(RootEvent::ModularProduct);
            o.event(RootEvent::MinimumScanProduct);
            let step = mul(root, root, r.q);
            let h = r.m >> 1;
            let mut current = root;
            let mut minimum = root;
            for visit in 0..h {
                o.event(RootEvent::MinimumScanVisit);
                minimum = minimum.min(current);
                if visit != h - 1 {
                    o.event(RootEvent::ModularProduct);
                    o.event(RootEvent::MinimumScanProduct);
                    current = mul(current, step, r.q);
                }
            }
            return Ok(minimum);
        }
    }
    Err(RootError::Exhausted)
}

#[cfg(test)]
#[path = "modular_tests.rs"]
mod modular_tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InverseError {
    NotCoprime,
    Capacity,
}

/// Bounded Euclid extracted from the clear Rns3 owner, unchanged numerically.
/// Callers supply odd 3 <= p < 2^61 and 0 <= a < p. Stack-only,
/// variable-time public arithmetic; no full-u64 or even-modulus contract.
pub(crate) fn inverse(a: u64, p: u64) -> Result<u64, InverseError> {
    let (mut r, mut s, mut x, mut y) = (i128::from(p), i128::from(a), 0_i128, 1_i128);
    for _ in 0..128 {
        if s == 0 {
            return if r == 1 {
                Ok(x.rem_euclid(i128::from(p)) as u64)
            } else {
                Err(InverseError::NotCoprime)
            };
        }
        let d = r / s;
        (r, s) = (s, r - d * s);
        (x, y) = (y, x - d * y);
    }
    Err(InverseError::Capacity)
}

/// Private arithmetic domain, not a key, security parameter set, or owner.
/// Public setup accepts odd composite moduli too. Every value operand of
/// encode/decode/add/sub/mul must be canonical below q. Fixed source patterns
/// do not certify target-machine constant time or register/copy erasure.
pub(crate) struct Montgomery64 {
    q: u64,
    qinv: u64,
    r2: u64,
}

impl Montgomery64 {
    /// PUBLIC modulus only. Six wrapping Newton steps double the inverse's
    /// correct low bits from one to 64. R and R^2 setup use public remainders.
    pub(crate) fn new(q: u64) -> Option<Self> {
        if !(3..(1_u64 << 61)).contains(&q) || q & 1 == 0 {
            return None;
        }
        let mut inverse = 1_u64;
        for _ in 0..6 {
            inverse = inverse.wrapping_mul(2_u64.wrapping_sub(q.wrapping_mul(inverse)));
        }
        let r = (1_u128 << 64) % u128::from(q);
        let r2 = (r * r % u128::from(q)) as u64;
        Some(Self { q, qinv: 0_u64.wrapping_sub(inverse), r2 })
    }

    fn subtract_once(&self, value: u64) -> u64 {
        let (reduced, borrow) = value.overflowing_sub(self.q);
        let mask = 0_u64.wrapping_sub(u64::from(borrow));
        (value & mask) | (reduced & !mask)
    }

    /// INTERNAL bounded REDC only: T<q^2, never an arbitrary-u128 interface.
    /// m<2^64, mq<2^125, T<2^122, so the sum fits u128. The cancellation
    /// makes it divisible by R; v<q+q^2/R<2q needs one masked subtraction.
    fn reduce(&self, product: u128) -> u64 {
        let m = (product as u64).wrapping_mul(self.qinv);
        let sum = product.wrapping_add(u128::from(m).wrapping_mul(u128::from(self.q)));
        self.subtract_once((sum >> 64) as u64)
    }

    /// Ordinary canonical coefficient -> canonical Montgomery coefficient xR.
    pub(crate) fn encode(&self, value: u64) -> u64 {
        self.reduce(u128::from(value).wrapping_mul(u128::from(self.r2)))
    }

    /// Only the documented external boundary leaves the Montgomery domain.
    pub(crate) fn decode(&self, value: u64) -> u64 {
        self.reduce(u128::from(value))
    }

    pub(crate) fn mul(&self, lhs: u64, rhs: u64) -> u64 {
        self.reduce(u128::from(lhs).wrapping_mul(u128::from(rhs)))
    }

    pub(crate) fn add(&self, lhs: u64, rhs: u64) -> u64 {
        self.subtract_once(lhs.wrapping_add(rhs))
    }

    pub(crate) fn sub(&self, lhs: u64, rhs: u64) -> u64 {
        let (difference, borrow) = lhs.overflowing_sub(rhs);
        difference.wrapping_add(self.q & 0_u64.wrapping_sub(u64::from(borrow)))
    }

    /// Full scan, including values following an invalid coefficient.
    pub(crate) fn invalid(&self, values: &[u64]) -> u64 {
        let mut invalid = 0;
        for &value in values {
            invalid |= u64::from(!value.overflowing_sub(self.q).1);
        }
        invalid
    }
}
