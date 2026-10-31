//! Public, exact clear arithmetic in F_q[X]/(X^N+1), N <= 8192.
//! Degrees above 4096 require CoefficientRing::bounded with DegreeBound::Public8192.
//!
//! No HE backend, protected authority, profile, ledger, tariff, or prepayment.
//! Timing is public/value-dependent; this is NOT a constant-time primitive.
//! Caller owns all buffers; no allocation or scratch buffer is required. Safe
//! Rust requires output disjoint from inputs; immutable inputs may overlap.

use crate::coefficient_layout::CoefficientRing;
use crate::coefficient_layout::modular::{mul, sub};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidModulus,
    InvalidRoot,
    BufferLength,
    NonCanonical,
    /// Element is zero, even, or outside the canonical range [1, 2N).
    InvalidAutomorphism,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operation {
    Forward,
    Inverse,
    PointwiseAdd,
    PointwiseMul,
    Automorphism,
}

/// Conservative event bounds, NOT scalar credits, complete work, or memory quote.
/// No implicit pricing or sum is defined. Refusals are bounded by the same quote.
/// `checks` covers length/parameter/residue/MR comparisons and trial remainders,
/// not loop/index/control instructions or separately counted canonical-correction
/// comparisons (those are bundled into modular add/sub). `residue_checks` is a
/// subset of `checks`.
/// `moves` counts coefficient/descriptor
/// word stores (including initialization), not every local variable assignment;
/// swaps are separately counted, not expanded into moves. Modular add/sub each
/// includes canonical correction; multiplication includes its u128 reduction.
/// Exponentiations contain exponent_steps (binary iterations), whose square and
/// optional multiply are ALSO included in modular_multiplications: these are
/// nested categories, not independent prices. MR witness squarings are included
/// in modular_multiplications, outside exponent_steps. No ledger consumes this.
/// Automorphism counts only checks, residue visits and output stores. Its public
/// index arithmetic (per output: two products, one addition, one remainder,
/// one subtraction and one division) is excluded, like other index/control
/// instructions; these are NOT coefficient modular multiplications/reductions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LogicalOperations {
    pub checks: u64,
    pub residue_checks: u64,
    pub moves: u64,
    pub swaps: u64,
    pub modular_additions: u64,
    pub modular_subtractions: u64,
    pub modular_multiplications: u64,
    pub exponentiations: u64,
    pub exponent_steps: u64,
    pub primality_rounds: u64,
}

/// Immutable validated descriptor. Unused stage-root slots are zero.
/// Thirteen slots per direction add 16 declared payload bytes over twelve;
/// this is not a physical peak-memory or compiler-stack-layout claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NegacyclicNtt {
    ring: CoefficientRing,
    psi: u64,
    psi_inv: u64,
    n_inv: u64,
    forward_stage_roots: [u64; 13],
    inverse_stage_roots: [u64; 13],
}

fn pow(mut a: u64, mut exponent: u64, q: u64) -> u64 {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = mul(result, a, q);
        }
        a = mul(a, a, q);
        exponent >>= 1;
    }
    result
}

// Deterministic Miller–Rabin for the FULL u64 domain, not just small limbs.
fn prime(q: u64) -> bool {
    if q < 2 {
        return false;
    }
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if q % p == 0 {
            return q == p;
        }
    }
    let s = (q - 1).trailing_zeros();
    let d = (q - 1) >> s;
    for base in [2, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        let a = base % q;
        if a == 0 {
            continue;
        }
        let mut x = pow(a, d, q);
        if x == 1 || x == q - 1 {
            continue;
        }
        let mut passed = false;
        for _ in 1..s {
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

/// Includes parameter/refusal checks, all seven MR rounds, both root checks,
/// two Fermat inverses, and both roots for every stage. At most 37 powers;
/// each has <=64 binary steps, <=128 modular products. Witness squarings
/// contribute <=7*64 further products. Valid CoefficientRing bounds make all
/// inventory arithmetic bounded, including refusal and constructor setup.
/// At N=8192: 37 powers, 2368 binary steps, 5184 modular products. Checks
/// remain a bundled conservative envelope; 128 moves cover at most 83
/// initialized/stored coefficient/descriptor words, not local temporaries.
pub fn construction_operations(ring: CoefficientRing) -> LogicalOperations {
    let stages = u64::from(ring.degree().trailing_zeros());
    let powers = 11 + 2 * stages;
    LogicalOperations {
        checks: 4096,
        residue_checks: 1,
        moves: 128,
        modular_multiplications: powers * 128 + 7 * 64,
        exponentiations: powers,
        exponent_steps: powers * 64,
        primality_rounds: 7,
        ..LogicalOperations::default()
    }
}

impl NegacyclicNtt {
    pub fn new(ring: CoefficientRing, psi: u64) -> Result<Self, Error> {
        let q = ring.modulus();
        let n = ring.degree() as u64;
        if !(3..(1_u64 << 61)).contains(&q) || !prime(q) || (q - 1) % (2 * n) != 0 {
            return Err(Error::InvalidModulus);
        }
        if psi >= q {
            return Err(Error::NonCanonical);
        }
        if pow(psi, n, q) != q - 1 || pow(psi, 2 * n, q) != 1 {
            return Err(Error::InvalidRoot);
        }
        let psi_inv = pow(psi, q - 2, q);
        let n_inv = pow(n, q - 2, q);
        let mut forward_stage_roots = [0; 13];
        let mut inverse_stage_roots = [0; 13];
        for stage in 0..ring.degree().trailing_zeros() as usize {
            let len = 1_u64 << (stage + 1);
            forward_stage_roots[stage] = pow(psi, 2 * n / len, q);
            inverse_stage_roots[stage] = pow(psi_inv, 2 * n / len, q);
        }
        // Publish only after every validation and setup computation succeeds.
        Ok(Self { ring, psi, psi_inv, n_inv, forward_stage_roots, inverse_stage_roots })
    }

    pub fn ring(&self) -> CoefficientRing {
        self.ring
    }

    pub fn operations(&self, operation: Operation) -> LogicalOperations {
        let n = self.ring.degree() as u64;
        let butterflies = n / 2 * u64::from(self.ring.degree().trailing_zeros());
        let mut quote = LogicalOperations::default();
        match operation {
            Operation::Automorphism => {
                // Three element predicates, two lengths, and every residue.
                quote.checks = 5 + n;
                quote.residue_checks = n;
                quote.moves = n;
            }
            Operation::Forward | Operation::Inverse => {
                quote.checks = 2 + n;
                quote.residue_checks = n;
                quote.moves = 2 * n + 2 * butterflies;
                quote.swaps = n / 2;
                quote.modular_additions = butterflies;
                quote.modular_subtractions = butterflies;
                quote.modular_multiplications = if n == 1 { 0 } else {
                    2 * butterflies + n * if operation == Operation::Inverse { 3 } else { 2 }
                };
            }
            Operation::PointwiseAdd | Operation::PointwiseMul => {
                quote.checks = 3 + 2 * n;
                quote.residue_checks = 2 * n;
                quote.moves = n;
                if operation == Operation::PointwiseAdd {
                    quote.modular_additions = n;
                } else {
                    quote.modular_multiplications = n;
                }
            }
        }
        quote
    }

    fn validate(&self, inputs: &[&[u64]], out: &[u64]) -> Result<(), Error> {
        if out.len() != self.ring.degree()
            || inputs.iter().any(|input| input.len() != self.ring.degree())
        {
            return Err(Error::BufferLength);
        }
        // Scan EVERY residue in EVERY input, even after detecting a bad value.
        let mut invalid = false;
        for input in inputs {
            for &value in *input {
                invalid |= value >= self.ring.modulus();
            }
        }
        if invalid { Err(Error::NonCanonical) } else { Ok(()) }
    }

    /// Natural order: F[k] = sum_j input[j]*psi^((2*k+1)*j) mod q.
    /// Exact lengths/canonical inputs checked before any output store.
    pub fn forward_into(&self, input: &[u64], out: &mut [u64]) -> Result<(), Error> {
        self.transform(input, out, false)
    }

    /// Natural-order inverse, including N^-1 and inverse twist.
    pub fn inverse_into(&self, input: &[u64], out: &mut [u64]) -> Result<(), Error> {
        self.transform(input, out, true)
    }

    /// Natural-order spectrum of sigma_g(f) = f(X^g) modulo X^N+1.
    /// `input` is already in this descriptor's NTT domain; no transform occurs.
    /// Requires canonical odd 1 <= g < 2N (not normalization modulo 2N).
    /// Element, exact lengths, then ALL canonical residues are validated before
    /// any store. Safe Rust requires immutable input and mutable output disjoint.
    /// O(N), no allocation/scratch, coefficient arithmetic or root change.
    ///
    /// sigma_g(f)(psi^(2k+1)) = f(psi^(g*(2k+1))) gives the gather below.
    /// Validated N <= 8192 implies 2N <= 16384 and g*(2k+1) <=
    /// 16383^2 = 268402689 < 2^28. All index arithmetic fits u64 and the
    /// resulting index < N fits usize on supported targets. Odd factors give
    /// an odd nonzero remainder, so subtraction cannot underflow. Since N is
    /// a power of two, every accepted g is invertible modulo 2N.
    ///
    /// Public algebra only: transforming ciphertext components changes the
    /// secret domain to sigma_g(s). This is NOT a key switch, evaluation key,
    /// encrypted execution capability, or noise/scale/ring-chain transition.
    pub fn automorphism_into(
        &self,
        input: &[u64],
        g: u64,
        out: &mut [u64],
    ) -> Result<(), Error> {
        let two_n = 2 * self.ring.degree() as u64;
        if g == 0 || g >= two_n || g & 1 == 0 {
            return Err(Error::InvalidAutomorphism);
        }
        self.validate(&[input], out)?;
        for (k, value) in out.iter_mut().enumerate() {
            let source = ((g * (2 * k as u64 + 1)) % two_n - 1) / 2;
            *value = input[source as usize];
        }
        Ok(())
    }

    fn transform(&self, input: &[u64], out: &mut [u64], inverse: bool) -> Result<(), Error> {
        self.validate(&[input], out)?;
        out.copy_from_slice(input);
        let n = out.len();
        if n == 1 {
            return Ok(());
        }
        let q = self.ring.modulus();
        if !inverse {
            let mut twist = 1;
            for value in out.iter_mut() {
                *value = mul(*value, twist, q);
                twist = mul(twist, self.psi, q);
            }
        }
        let mut reversed = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while reversed & bit != 0 {
                reversed ^= bit;
                bit >>= 1;
            }
            reversed ^= bit;
            if i < reversed {
                out.swap(i, reversed);
            }
        }
        let roots = if inverse { &self.inverse_stage_roots } else { &self.forward_stage_roots };
        for (stage, &root) in roots[..n.trailing_zeros() as usize].iter().enumerate() {
            let len = 1 << (stage + 1);
            for start in (0..n).step_by(len) {
                let mut twiddle = 1;
                for j in 0..len / 2 {
                    let a = out[start + j];
                    let b = mul(out[start + j + len / 2], twiddle, q);
                    // Canonical a,b<q; sum<2q<2^62, product<2^122.
                    let sum = a + b;
                    out[start + j] = if sum >= q { sum - q } else { sum };
                    out[start + j + len / 2] = sub(a, b, q);
                    twiddle = mul(twiddle, root, q);
                }
            }
        }
        if inverse {
            let mut twist = 1;
            for value in out.iter_mut() {
                *value = mul(mul(*value, self.n_inv, q), twist, q);
                twist = mul(twist, self.psi_inv, q);
            }
        }
        Ok(())
    }

    pub fn pointwise_add_into(&self, a: &[u64], b: &[u64], out: &mut [u64]) -> Result<(), Error> {
        self.validate(&[a, b], out)?;
        let q = self.ring.modulus();
        for ((&a, &b), value) in a.iter().zip(b).zip(out) {
            let sum = a + b;
            *value = if sum >= q { sum - q } else { sum };
        }
        Ok(())
    }

    pub fn pointwise_mul_into(&self, a: &[u64], b: &[u64], out: &mut [u64]) -> Result<(), Error> {
        self.validate(&[a, b], out)?;
        for ((&a, &b), value) in a.iter().zip(b).zip(out) {
            *value = mul(a, b, self.ring.modulus());
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "coefficient_ntt/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "coefficient_ntt_automorphism_tests.rs"]
mod automorphism_tests;
