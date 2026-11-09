//! Ordinary unrun test definitions. Fixtures are public coefficients, never a
//! sampler, entropy source, distribution, adopted profile, or security claim.
//! BigInt expected sides share neither production arithmetic nor scheduling.

use super::fixed::{self, Error, Operation};
use super::{NegacyclicNtt, schedule};
use crate::coefficient_layout::modular::Montgomery64;
use crate::coefficient_layout::{CoefficientRing, DegreeBound};
use crate::kernel_support::KernelWorkLimit;
use num_bigint::BigInt;

fn residue(value: BigInt, q: u64) -> u64 {
    let modulus = BigInt::from(q);
    let reduced = ((value % &modulus) + &modulus) % &modulus;
    let (_, words) = reduced.to_u64_digits();
    words.first().copied().unwrap_or(0)
}

fn power(value: u64, exponent: u64, q: u64) -> u64 {
    residue(BigInt::from(value).modpow(&BigInt::from(exponent), &BigInt::from(q)), q)
}

// Euclid works for odd composite q too; do not assume a prime in the REDC oracle.
fn inverse(value: BigInt, q: u64) -> BigInt {
    let (mut r, mut next_r) = (BigInt::from(q), value);
    let (mut x, mut next_x) = (BigInt::from(0), BigInt::from(1));
    while next_r != BigInt::from(0) {
        let quotient = &r / &next_r;
        (r, next_r) = (next_r.clone(), r - &quotient * next_r);
        (x, next_x) = (next_x.clone(), x - &quotient * next_x);
    }
    assert_eq!(r, BigInt::from(1));
    BigInt::from(residue(x, q))
}

fn plan(n: usize, q: u64, psi: u64) -> NegacyclicNtt {
    assert_eq!(power(psi, n as u64, q), q - 1);
    assert_eq!(power(psi, 2 * n as u64, q), 1);
    NegacyclicNtt::new(CoefficientRing::new(n, q).unwrap(), psi).unwrap()
}

fn direct(input: &[u64], psi: u64, q: u64, inverse: bool) -> Vec<u64> {
    let n = input.len();
    let root = if inverse { power(psi, q - 2, q) } else { psi };
    let inverse_n = power(n as u64, q - 2, q);
    (0..n)
        .map(|at| {
            let mut sum = BigInt::from(0);
            for (from, &value) in input.iter().enumerate() {
                let exponent = if inverse { (2 * from + 1) * at } else { (2 * at + 1) * from };
                sum += BigInt::from(value) * BigInt::from(power(root, exponent as u64, q));
            }
            if inverse {
                sum *= BigInt::from(inverse_n);
            }
            residue(sum, q)
        })
        .collect()
}

fn convolution(lhs: &[u64], rhs: &[u64], q: u64) -> Vec<u64> {
    let n = lhs.len();
    let mut whole = vec![BigInt::from(0); n];
    for (i, &a) in lhs.iter().enumerate() {
        for (j, &b) in rhs.iter().enumerate() {
            let term = BigInt::from(a) * BigInt::from(b);
            if i + j >= n {
                whole[i + j - n] -= term;
            } else {
                whole[i + j] += term;
            }
        }
    }
    whole.into_iter().map(|value| residue(value, q)).collect()
}

fn work(plan: &NegacyclicNtt, operation: Operation) -> KernelWorkLimit {
    KernelWorkLimit::new(fixed::quote(plan, operation).unwrap().total())
}

#[test]
fn montgomery_redc_encode_decode_and_boundary_bigint_oracles() {
    let r = BigInt::from(1) << 64_usize;
    for q in [3, 17, 257, 4095, (1_u64 << 61) - 1] {
        let domain = Montgomery64::new(q).unwrap();
        let inverse_r = inverse(r.clone(), q);
        let values = [0, 1, q / 2, q - 2, q - 1];
        for a in values {
            let encoded = domain.encode(a);
            assert!(encoded < q);
            assert_eq!(encoded, residue(BigInt::from(a) * &r, q));
            assert_eq!(domain.decode(a), residue(BigInt::from(a) * &inverse_r, q));
            assert_eq!(domain.decode(encoded), a);
            for b in values {
                // REDC is deliberately not exposed as an unchecked u128 API.
                // mul/encode/decode exercise its bounded canonical products.
                assert_eq!(
                    domain.mul(a, b),
                    residue(BigInt::from(a) * BigInt::from(b) * &inverse_r, q)
                );
                let left = domain.encode(a);
                let right = domain.encode(b);
                assert_eq!(
                    domain.decode(domain.mul(left, right)),
                    residue(BigInt::from(a) * BigInt::from(b), q)
                );
                assert_eq!(
                    domain.decode(domain.add(left, right)),
                    residue(BigInt::from(a) + BigInt::from(b), q)
                );
                assert_eq!(
                    domain.decode(domain.sub(left, right)),
                    residue(BigInt::from(a) - BigInt::from(b), q)
                );
            }
        }
        assert_eq!(domain.invalid(&values), 0);
        assert_ne!(domain.invalid(&[q, 0, q - 1, u64::MAX]), 0);
        assert_ne!(domain.invalid(&[0, 1, q - 1, q]), 0);
    }
    for q in [0, 1, 2, 4, 1_u64 << 61, u64::MAX] {
        assert!(Montgomery64::new(q).is_none());
    }
}

#[test]
fn natural_order_inverse_roundtrip_and_signed_products_bigint() {
    // M_61 is the existing near-limit prime fixture, degree ONE only. No new
    // root/prime search or nontrivial near-limit root assertion is introduced.
    let large = (1_u64 << 61) - 1;
    for (n, q, psi) in [
        (1, 17, 16),
        (2, 17, 4),
        (4, 17, 2),
        (8, 17, 3),
        (16, 97, 28),
        (1, large, large - 1),
    ] {
        let plan = plan(n, q, psi);
        let mut high = vec![0; n];
        high[n - 1] = q - 1;
        let listed = [q - 1, 1, q - 2, 0, 1, q - 1, 0, q - 2];
        let cases = [
            vec![0; n],
            vec![q - 1; n],
            high,
            (0..n).map(|i| listed[i % listed.len()]).collect(),
        ];
        for input in &cases {
            let original = input.clone();
            let mut forward = vec![99; n];
            let mut inverse = vec![99; n];
            let mut public = vec![99; n];
            fixed::forward_into(&plan, input, &mut forward, &mut work(&plan, Operation::Forward))
                .unwrap();
            assert_eq!(forward, direct(input, psi, q, false));
            plan.forward_into(input, &mut public).unwrap();
            assert_eq!(forward, public);
            // Arbitrary spectrum: inverse oracle is not merely a roundtrip.
            fixed::inverse_into(&plan, input, &mut inverse, &mut work(&plan, Operation::Inverse))
                .unwrap();
            assert_eq!(inverse, direct(input, psi, q, true));
            plan.inverse_into(input, &mut public).unwrap();
            assert_eq!(inverse, public);
            fixed::inverse_into(
                &plan,
                &forward,
                &mut inverse,
                &mut work(&plan, Operation::Inverse),
            )
            .unwrap();
            assert_eq!(inverse, *input);
            for rhs in &cases {
                let mut output = vec![99; n];
                let mut scratch = vec![99; 2 * n];
                let mut budget = work(&plan, Operation::Product);
                fixed::negacyclic_product_into(
                    &plan,
                    input,
                    rhs,
                    &mut output,
                    &mut scratch,
                    &mut budget,
                )
                .unwrap();
                assert_eq!(output, convolution(input, rhs, q));
                assert_eq!(budget.remaining(), 0);
                assert_eq!(scratch, vec![0; 2 * n]);
            }
            // Same immutable input in both positions must remain supported.
            let mut output = vec![99; n];
            let mut scratch = vec![99; 2 * n];
            fixed::negacyclic_product_into(
                &plan,
                input,
                input,
                &mut output,
                &mut scratch,
                &mut work(&plan, Operation::Product),
            )
            .unwrap();
            assert_eq!(output, convolution(input, input, q));
            assert_eq!(scratch, vec![0; 2 * n]);
            assert_eq!(*input, original);
        }
    }
}

#[test]
fn existing_plan_rejects_malformed_moduli_and_roots() {
    for (n, q, psi) in [(1, 9, 8), (16, 17, 3)] {
        let ring = CoefficientRing::new(n, q).unwrap();
        assert_eq!(NegacyclicNtt::new(ring, psi), Err(super::Error::InvalidModulus));
    }
    let ring = CoefficientRing::new(8, 17).unwrap();
    for psi in [0, 1, 2, 4, 16] {
        assert_eq!(NegacyclicNtt::new(ring, psi), Err(super::Error::InvalidRoot));
    }
    for psi in [17, u64::MAX] {
        assert_eq!(NegacyclicNtt::new(ring, psi), Err(super::Error::NonCanonical));
    }
    assert!(NegacyclicNtt::new(ring, 3).is_ok());
}

#[test]
fn actual_public_8192_degree_rejected_before_geometry_work_or_values() {
    let ring = CoefficientRing::bounded(8192, 65537, DegreeBound::Public8192).unwrap();
    let plan = NegacyclicNtt::new(ring, 81).unwrap();
    let mut output = [99];
    let mut scratch = [98; 2];
    let mut budget = KernelWorkLimit::new(0);
    for operation in [Operation::Forward, Operation::Inverse, Operation::Product] {
        assert_eq!(fixed::quote(&plan, operation), Err(Error::Degree));
        let result = match operation {
            Operation::Forward => fixed::forward_into(&plan, &[65537], &mut output, &mut budget),
            Operation::Inverse => fixed::inverse_into(&plan, &[65537], &mut output, &mut budget),
            Operation::Product => fixed::negacyclic_product_into(
                &plan,
                &[65537],
                &[u64::MAX],
                &mut output,
                &mut scratch,
                &mut budget,
            ),
        };
        assert_eq!(result, Err(Error::Degree));
        assert_eq!(output, [99]);
        assert_eq!(scratch, [98; 2]);
        assert_eq!(budget.remaining(), 0);
    }
    // The selector is NOT a secret opt-in. Conversely an actual legacy degree
    // is admissible even when its descriptor used the explicit public selector.
    let ring = CoefficientRing::bounded(4096, 65537, DegreeBound::Public8192).unwrap();
    let plan = NegacyclicNtt::new(ring, 6561).unwrap();
    for operation in [Operation::Forward, Operation::Inverse, Operation::Product] {
        assert!(fixed::quote(&plan, operation).is_ok());
    }
}

struct Buffers {
    lhs: Vec<u64>,
    rhs: Vec<u64>,
    output: Vec<u64>,
    scratch: Vec<u64>,
}

impl Buffers {
    fn new() -> Self {
        Self {
            lhs: vec![0, 1, 2, 3, 4, 5, 15, 16],
            rhs: vec![16, 15, 5, 4, 3, 2, 1, 0],
            output: vec![99; 8],
            scratch: vec![98; 16],
        }
    }

    fn call(
        &mut self,
        plan: &NegacyclicNtt,
        operation: Operation,
        budget: &mut KernelWorkLimit,
    ) -> Result<(), Error> {
        match operation {
            Operation::Forward => fixed::forward_into(plan, &self.lhs, &mut self.output, budget),
            Operation::Inverse => fixed::inverse_into(plan, &self.lhs, &mut self.output, budget),
            Operation::Product => fixed::negacyclic_product_into(
                plan,
                &self.lhs,
                &self.rhs,
                &mut self.output,
                &mut self.scratch,
                budget,
            ),
        }
    }
}

#[test]
fn every_shape_refusal_preserves_buffers_and_counter() {
    let plan = plan(8, 17, 3);
    for operation in [Operation::Forward, Operation::Inverse, Operation::Product] {
        let slots = if operation == Operation::Product { 4 } else { 2 };
        for slot in 0..slots {
            let expected = if operation == Operation::Product && slot == 3 { 16 } else { 8 };
            for length in [0, expected - 1, expected + 1] {
                let mut buffers = Buffers::new();
                // Bad values elsewhere ensure geometry takes precedence.
                buffers.lhs.fill(17);
                buffers.rhs.fill(u64::MAX);
                let values = match (operation, slot) {
                    (_, 0) => &mut buffers.lhs,
                    (Operation::Product, 1) => &mut buffers.rhs,
                    (Operation::Product, 2) => &mut buffers.output,
                    (Operation::Product, 3) => &mut buffers.scratch,
                    (_, 1) => &mut buffers.output,
                    _ => panic!("invalid public shape slot"),
                };
                values.resize(length, 99);
                let output = buffers.output.clone();
                let scratch = buffers.scratch.clone();
                let mut budget = work(&plan, operation);
                let before = budget.remaining();
                assert_eq!(buffers.call(&plan, operation, &mut budget), Err(Error::Geometry));
                assert_eq!(buffers.output, output);
                assert_eq!(buffers.scratch, scratch);
                assert_eq!(budget.remaining(), before);
            }
        }
    }
}

#[test]
fn all_input_arrays_late_bad_values_spend_work_without_writes() {
    let plan = plan(8, 17, 3);
    for operation in [Operation::Forward, Operation::Inverse, Operation::Product] {
        let slots = if operation == Operation::Product { 2 } else { 1 };
        for slot in 0..slots {
            for at in [0, 7] {
                for value in [17, u64::MAX] {
                    let mut buffers = Buffers::new();
                    if slot == 0 {
                        buffers.lhs[at] = value;
                    } else {
                        buffers.rhs[at] = value;
                    }
                    let lhs = buffers.lhs.clone();
                    let rhs = buffers.rhs.clone();
                    let output = buffers.output.clone();
                    let scratch = buffers.scratch.clone();
                    let mut budget = work(&plan, operation);
                    assert_eq!(
                        buffers.call(&plan, operation, &mut budget),
                        Err(Error::NonCanonical)
                    );
                    assert_eq!(budget.remaining(), 0);
                    assert_eq!(buffers.output, output);
                    assert_eq!(buffers.scratch, scratch);
                    assert_eq!(buffers.lhs, lhs);
                    assert_eq!(buffers.rhs, rhs);
                }
            }
        }
    }
    // Both arrays invalid: production source still scans the entire second
    // array; this test checks the fixed error/state contract, not read tracing.
    let mut buffers = Buffers::new();
    buffers.lhs[0] = 17;
    buffers.rhs[7] = u64::MAX;
    let mut budget = work(&plan, Operation::Product);
    assert_eq!(buffers.call(&plan, Operation::Product, &mut budget), Err(Error::NonCanonical));
    assert_eq!(buffers.output, vec![99; 8]);
    assert_eq!(buffers.scratch, vec![98; 16]);
    assert_eq!(budget.remaining(), 0);
}

#[test]
fn exact_one_short_and_exhausted_quotes_include_cleanup() {
    let plan = plan(8, 17, 3);
    for (operation, expected) in [
        (Operation::Forward, 187),
        (Operation::Inverse, 196),
        (Operation::Product, 502),
    ] {
        let quote = fixed::quote(&plan, operation).unwrap();
        assert_eq!(quote.total(), expected);
        assert_eq!(quote.public_setup, 14);
        assert_eq!(quote.butterflies, if operation == Operation::Product { 216 } else { 72 });
        assert_eq!(quote.cleanup, if operation == Operation::Product { 24 } else { 8 });
        let mut buffers = Buffers::new();
        // Work refusal precedes even an invalid value, preserving all state.
        buffers.lhs[7] = 17;
        let mut short = KernelWorkLimit::new(expected - 1);
        assert_eq!(buffers.call(&plan, operation, &mut short), Err(Error::Work));
        assert_eq!(short.remaining(), expected - 1);
        assert_eq!(buffers.output, vec![99; 8]);
        assert_eq!(buffers.scratch, vec![98; 16]);
        buffers.lhs[7] = 16;
        let mut exact = KernelWorkLimit::new(expected);
        buffers.call(&plan, operation, &mut exact).unwrap();
        assert_eq!(exact.remaining(), 0);
        assert_eq!(buffers.scratch, vec![if operation == Operation::Product { 0 } else { 98 }; 16]);
        let output = buffers.output.clone();
        buffers.scratch.fill(98);
        assert_eq!(buffers.call(&plan, operation, &mut exact), Err(Error::Work));
        assert_eq!(buffers.output, output);
        assert_eq!(buffers.scratch, vec![98; 16]);
        assert_eq!(exact.remaining(), 0);
    }
    for (n, psi, forward, inverse, product) in [
        (1, 16, 20, 20, 27),
        (2, 4, 47, 50, 106),
        (4, 2, 90, 95, 227),
    ] {
        let plan = self::plan(n, 17, psi);
        for (operation, expected) in [
            (Operation::Forward, forward),
            (Operation::Inverse, inverse),
            (Operation::Product, product),
        ] {
            assert_eq!(fixed::quote(&plan, operation).unwrap().total(), expected);
        }
    }
}

#[test]
fn checked_inventory_overflow_is_not_a_degree_opt_in() {
    for operation in [Operation::Forward, Operation::Inverse, Operation::Product] {
        assert_eq!(fixed::inventory(operation, u64::MAX, 64), Err(Error::Overflow));
        assert_eq!(fixed::inventory(operation, 8, u64::MAX), Err(Error::Overflow));
        assert_eq!(fixed::inventory(operation, u64::MAX / 4, 0), Err(Error::Overflow));
    }
}

// Source-event traces, not runtime instrumentation or machine-time evidence.
// Event 0=swap; 1/2=forward value/twist products; 3=butterfly product;
// 4/5=add/sub; 6=twiddle product; 7/8/9=inverse scale/value/twist products.
type Event = (u8, usize, usize, usize);

fn butterfly_events(events: &mut Vec<Event>, stage: usize, low: usize, high: usize) {
    for kind in [3, 4, 5, 6] {
        events.push((kind, stage, low, high));
    }
}

fn traversal_events(n: usize, inverse: bool, shared: bool) -> Vec<Event> {
    let mut events = Vec::new();
    if n == 1 {
        return events;
    }
    if !inverse {
        for i in 0..n {
            events.push((1, i, 0, 0));
            events.push((2, i, 0, 0));
        }
    }
    if shared {
        for (low, high) in schedule::swaps(n) {
            events.push((0, low, high, 0));
        }
        for stage in schedule::stages(n) {
            for start in stage.blocks() {
                for (low, high) in stage.butterflies(start) {
                    butterfly_events(&mut events, stage.index, low, high);
                }
            }
        }
    } else {
        // Independent closed-form reversal, then original stage/block/j loops.
        for i in 1..n {
            let reversed = i.reverse_bits() >> (usize::BITS - n.trailing_zeros());
            if i < reversed {
                events.push((0, i, reversed, 0));
            }
        }
        for stage in 0..n.trailing_zeros() as usize {
            let len = 1 << (stage + 1);
            for start in (0..n).step_by(len) {
                for j in 0..len / 2 {
                    butterfly_events(&mut events, stage, start + j, start + j + len / 2);
                }
            }
        }
    }
    if inverse {
        for i in 0..n {
            events.push((7, i, 0, 0));
            events.push((8, i, 0, 0));
            events.push((9, i, 0, 0));
        }
    }
    events
}

#[test]
fn exhaustive_small_schedule_and_legacy_arithmetic_event_order() {
    for (n, q, psi) in [(1, 17, 16), (2, 17, 4), (4, 17, 2), (8, 17, 3), (16, 97, 28)] {
        let plan = plan(n, q, psi);
        for inverse in [false, true] {
            let actual = traversal_events(n, inverse, true);
            assert_eq!(actual, traversal_events(n, inverse, false));
            let operation = if inverse {
                super::Operation::Inverse
            } else {
                super::Operation::Forward
            };
            let quote = plan.operations(operation);
            let products = actual
                .iter()
                .filter(|&&(kind, _, _, _)| matches!(kind, 1 | 2 | 3 | 6 | 7 | 8 | 9))
                .count() as u64;
            assert_eq!(products, quote.modular_multiplications);
            let additions = actual.iter().filter(|&&(kind, _, _, _)| kind == 4).count() as u64;
            let subtractions = actual.iter().filter(|&&(kind, _, _, _)| kind == 5).count() as u64;
            assert_eq!(additions, quote.modular_additions);
            assert_eq!(subtractions, quote.modular_subtractions);
            let swaps = actual.iter().filter(|&&(kind, _, _, _)| kind == 0).count() as u64;
            assert!(swaps <= quote.swaps);
        }
        for stage in schedule::stages(n) {
            let mut visits = vec![0; n];
            for start in stage.blocks() {
                for (low, high) in stage.butterflies(start) {
                    visits[low] += 1;
                    visits[high] += 1;
                }
            }
            assert_eq!(visits, vec![1; n]);
        }
    }
}
