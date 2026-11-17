use super::*;
use crate::coefficient_layout::{CoefficientRing, LayoutError};
use crate::coefficient_ntt::{Error as NttError, NegacyclicNtt, construction_operations};

// Independent small-field oracle: trial division and repeated multiplication,
// not the shared primality, binary exponentiation, or odd-power scan.
fn small_prime(q: u64) -> bool {
    q >= 2 && !(2..q).any(|d| q % d == 0)
}

fn small_power(a: u64, e: u64, q: u64) -> u64 {
    (0..e).fold(1, |v, _| v * a % q)
}

fn request(q: u64, m: u64, candidate_max: u64) -> RootRequest {
    RootRequest { q, m, candidate_max }
}

#[derive(Default)]
struct Counts([u64; 12]);

impl RootObserver for Counts {
    fn event(&mut self, event: RootEvent) {
        self.0[event as usize] += 1;
    }
}

fn bounded(counts: &Counts, quote: RootWorkQuote) {
    let bounds = [
        quote.exponentiations, quote.exponent_steps, quote.mr_squarings,
        quote.modular_products, quote.trial_remainders, quote.base_reductions,
        quote.candidate_iterations, quote.minimum_scan_visits,
        quote.validation_predicates, quote.validation_remainders,
        quote.projection_divisions, quote.minimum_scan_visits,
    ];
    for (actual, bound) in counts.0.into_iter().zip(bounds) {
        assert!(actual <= bound, "logical count {actual} exceeds {bound}");
    }
}

#[test]
fn public_root_small_prime_canonical_oracle() {
    for q in (3..=97).filter(|&q| small_prime(q)) {
        for m in [2, 4, 8, 16, 32, 64] {
            if (q - 1) % m != 0 { continue; }
            let minimum = (1..q).find(|&r| {
                small_power(r, m, q) == 1 && small_power(r, m / 2, q) == q - 1
            }).unwrap();
            for candidate_max in 2..q {
                let found = (2..=candidate_max).any(|a| {
                    let r = small_power(a, (q - 1) / m, q);
                    small_power(r, m, q) == 1 && small_power(r, m / 2, q) == q - 1
                });
                assert_eq!(canonical_power_two_root(request(q, m, candidate_max)),
                    if found { Ok(minimum) } else { Err(RootError::Exhausted) });
            }
        }
    }
}

#[test]
fn public_root_composites_and_zero_witness() {
    for q in [9, 25, 49, 561, 1105, 1729, 2047, 1_373_653,
        3_215_031_751, 341_550_071_728_321, 3_825_123_056_546_413_051]
    {
        assert!(!prime(q));
        assert_eq!(canonical_power_two_root(request(q, 2, 2)), Err(RootError::Composite));
    }
    for q in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        assert!(prime(q));
    }
    // Prime divisors of witness constants, including primes past trial division.
    for q in [73, 193, 407_521, 299_210_837] {
        assert!(prime(q));
        let mut counts = Counts::default();
        assert!(observed_prime(q, &mut counts));
        assert_eq!(counts.0[RootEvent::BaseReduction as usize], 7);
        assert!(counts.0[RootEvent::Exponentiation as usize] < 7);
    }
    // 14089 = 73*193 divides witness 28178: earlier witnesses must still fail.
    assert!(!prime(14_089));
    assert!(!prime(0));
    assert!(!prime(1));
}

#[test]
fn public_root_invalid_modulus_and_order() {
    for q in [0, 1, 2, 4, 6, u64::MAX - 1] {
        assert_eq!(canonical_power_two_root(request(q, 0, 0)), Err(RootError::InvalidModulus));
    }
    for m in [0, 1, 3, 6, 32, u64::MAX, 1_u64 << 63] {
        assert_eq!(canonical_power_two_root(request(17, m, 0)), Err(RootError::InvalidOrder));
    }
    assert_eq!(public_root_work_quote(request(9, 16, 0)), Err(RootError::InvalidOrder));
}

#[test]
fn public_root_candidate_bounds_inclusive() {
    for candidate_max in [0, 1, 17, 18, u64::MAX] {
        assert_eq!(canonical_power_two_root(request(17, 4, candidate_max)),
            Err(RootError::InvalidCandidateRange));
    }
    assert_eq!(canonical_power_two_root(request(17, 4, 2)), Err(RootError::Exhausted));
    assert_eq!(canonical_power_two_root(request(17, 4, 3)), Ok(4));
    assert_eq!(canonical_power_two_root(request(17, 4, 16)), Ok(4));
    assert_eq!(canonical_power_two_root(request(3, 2, 2)), Ok(2));
}

#[test]
fn public_root_exhaustion_no_fallback() {
    let r = request(17, 8, 2);
    let mut counts = Counts::default();
    assert_eq!(observed_root(r, &mut counts), Err(RootError::Exhausted));
    assert_eq!(counts.0[RootEvent::CandidateIteration as usize], 1);
    assert_eq!(counts.0[RootEvent::MinimumScanVisit as usize], 0);
    bounded(&counts, public_root_work_quote(r).unwrap());
}

#[test]
fn public_root_full_width_and_ntt_narrow_regression() {
    let q = 18_446_744_073_709_551_557;
    assert_eq!(mul(q - 1, q - 1, q), 1);
    assert_eq!(mul(u64::MAX - 1, u64::MAX - 1, u64::MAX), 1);
    assert_eq!(mul(u64::MAX, u64::MAX, u64::MAX), 0);
    assert_eq!(canonical_power_two_root(request(q, 2, 2)), Ok(q - 1));
    assert_eq!(CoefficientRing::new(2, q), Err(LayoutError::InvalidModulus));
    assert_eq!(CoefficientRing::new(2, 1_u64 << 61), Err(LayoutError::InvalidModulus));
    for modulus in [9, 7] {
        assert_eq!(NegacyclicNtt::new(CoefficientRing::new(2, modulus).unwrap(), 99),
            Err(NttError::InvalidModulus));
    }
    let ring = CoefficientRing::new(2, 17).unwrap();
    assert_eq!(NegacyclicNtt::new(ring, 17), Err(NttError::NonCanonical));
    assert_eq!(NegacyclicNtt::new(ring, 1), Err(NttError::InvalidRoot));
    let plan = NegacyclicNtt::new(ring, 4).unwrap();
    let mut spectrum = [0; 2];
    let mut restored = [0; 2];
    plan.forward_into(&[3, 5], &mut spectrum).unwrap();
    assert_eq!(spectrum, [6, 0]);
    plan.inverse_into(&spectrum, &mut restored).unwrap();
    assert_eq!(restored, [3, 5]);
    let quote = construction_operations(ring);
    assert_eq!((quote.exponentiations, quote.exponent_steps, quote.modular_multiplications),
        (13, 832, 2112));
}

#[test]
fn public_root_quote_overflow_before_primality() {
    for r in [request(u64::MAX, 2, u64::MAX - 1),
        request(u64::MAX, 2, 1_u64 << 56)]
    {
        let mut counts = Counts::default();
        assert_eq!(public_root_work_quote(r), Err(RootError::QuoteOverflow));
        assert_eq!(observed_root(r, &mut counts), Err(RootError::QuoteOverflow));
        for event in [RootEvent::TrialRemainder, RootEvent::BaseReduction,
            RootEvent::Exponentiation, RootEvent::CandidateIteration, RootEvent::ProjectionDivision]
        {
            assert_eq!(counts.0[event as usize], 0);
        }
    }
}

#[test]
fn public_root_measured_logical_bounds() {
    for r in [request(3, 2, 2), request(17, 2, 3), request(17, 4, 3),
        request(97, 32, 96), request(9, 2, 8), request(17, 8, 2),
        request(18_446_744_073_709_551_557, 2, 2),
        request((1_u64 << 63) + 1, 1_u64 << 63, 2)]
    {
        let quote = public_root_work_quote(r).unwrap();
        let c = r.candidate_max - 1;
        let p = 7 + 3 * c;
        assert_eq!(quote.exponentiations, p);
        assert_eq!(quote.exponent_steps, 64 * p);
        assert_eq!(quote.mr_squarings, 441);
        assert_eq!(quote.modular_products, 128 * p + 441 + (r.m >> 1));
        assert_eq!((quote.trial_remainders, quote.base_reductions), (12, 7));
        assert_eq!((quote.candidate_iterations, quote.minimum_scan_visits), (c, r.m >> 1));
        assert_eq!((quote.validation_predicates, quote.validation_remainders,
            quote.projection_divisions), (7, 1, 1));
        assert_eq!((quote.heap_bytes, quote.recursion_depth), (0, 0));
        assert_eq!(quote.auxiliary_storage, RootAuxiliaryStorage::Constant);
        let mut counts = Counts::default();
        let result = observed_root(r, &mut counts);
        bounded(&counts, quote);
        if r.m == 2 && result.is_ok() {
            assert_eq!(counts.0[RootEvent::MinimumScanVisit as usize], 1);
            assert_eq!(counts.0[RootEvent::MinimumScanProduct as usize], 1);
        }
    }
}
