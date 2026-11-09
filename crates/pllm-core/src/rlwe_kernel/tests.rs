//! Fixed, enumerated coefficients only; no sampler or probabilistic assertion.
//! BigInt oracle uses whole signed convolution, Euclidean reduction, and an
//! exact positive rational half-up quotient, never production arithmetic helpers.

use super::arithmetic as ar;
use super::bfv::{add_into, decrypt_into, encrypt_into, key_equation_into};
use super::*;
use crate::kernel_support::WorkExhausted;
use num_bigint::BigInt;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn supports(value: u64) -> Supports {
    Supports {
        s: value,
        u: value,
        e: value,
        e0: value,
        e1: value,
    }
}

fn residue(value: BigInt, modulus: u64) -> u64 {
    let modulus = BigInt::from(modulus);
    let reduced = ((value % &modulus) + &modulus) % &modulus;
    let (_, digits) = reduced.to_u64_digits();
    digits.first().copied().unwrap_or(0)
}

fn signed_words(values: &[i64]) -> Vec<BigInt> {
    values.iter().copied().map(BigInt::from).collect()
}

fn unsigned_words(values: &[u64]) -> Vec<BigInt> {
    values.iter().copied().map(BigInt::from).collect()
}

fn whole_convolution(lhs: &[BigInt], rhs: &[BigInt]) -> Vec<BigInt> {
    let n = lhs.len();
    let mut result = vec![BigInt::from(0); n];
    for i in 0..n {
        for j in 0..n {
            let term = &lhs[i] * &rhs[j];
            if i + j >= n {
                result[i + j - n] -= term;
            } else {
                result[i + j] += term;
            }
        }
    }
    result
}

fn oracle_key(s: &[i64], a: &[u64], e: &[i64], q: u64) -> Vec<u64> {
    let product = whole_convolution(&unsigned_words(a), &signed_words(s));
    let mut result: Vec<u64> = product
        .iter()
        .zip(e)
        .map(|(v, &error)| residue(BigInt::from(error) - v, q))
        .collect();
    result.extend_from_slice(a);
    result
}

fn oracle_encrypt(
    pk: &[u64],
    message: &[u64],
    u: &[i64],
    e0: &[i64],
    e1: &[i64],
    q: u64,
    t: u64,
) -> Vec<u64> {
    let n = message.len();
    let first = whole_convolution(&unsigned_words(&pk[..n]), &signed_words(u));
    let second = whole_convolution(&unsigned_words(&pk[n..]), &signed_words(u));
    let delta = BigInt::from(q) / BigInt::from(t);
    let mut result = Vec::new();
    for i in 0..n {
        result.push(residue(&first[i] + BigInt::from(e0[i]) + &delta * message[i], q));
    }
    for i in 0..n {
        result.push(residue(&second[i] + BigInt::from(e1[i]), q));
    }
    result
}

fn oracle_round(phase: u64, q: u64, t: u64) -> u64 {
    let numerator = BigInt::from(2) * BigInt::from(t) * BigInt::from(phase) + BigInt::from(q);
    let denominator = BigInt::from(2) * BigInt::from(q);
    residue(numerator / denominator, t)
}

fn oracle_decrypt(s: &[i64], ct: &[u64], q: u64, t: u64) -> Vec<u64> {
    let n = s.len();
    let product = whole_convolution(&unsigned_words(&ct[n..]), &signed_words(s));
    (0..n)
        .map(|i| oracle_round(residue(&product[i] + BigInt::from(ct[i]), q), q, t))
        .collect()
}

fn exact_work(operation: Operation, n: usize) -> KernelWorkLimit {
    KernelWorkLimit::new(quote(operation, n).unwrap())
}

/// Six fixed pipelines; each checks key, two encryptions, addition, and three
/// decryptions against independent equations, then checks plaintext correctness.
#[test]
fn fixed_equations_and_one_add_bigint() {
    let cases = [
        (1, 257, 2),
        (2, 257, 2),
        (4, 257, 2),
        (8, 257, 2),
        (4, 4095, 16),
        (4, (1_u64 << 61) - 1, 16),
    ];
    // Repeated fixed lists are coefficient vectors, not a distribution law.
    let secret = [-1, 1, 0, -1, 1, 0, 1, -1];
    let ephemeral = [1, -1, 1, 0, -1, 1, 0, -1];
    let error = [-1, 1, -1, 1, 0, -1, 1, 0];
    let error0 = [1, -1, 0, 1, -1, 0, 1, -1];
    let error1 = [-1, 0, 1, -1, 1, -1, 0, 1];
    for (n, q, t) in cases {
        let params = Params::new(n, q, t, supports(1)).unwrap();
        let s = &secret[..n];
        let u = &ephemeral[..n];
        let e = &error[..n];
        let e0 = &error0[..n];
        let e1 = &error1[..n];
        let a_list = [q - 1, 0, 1, q - 2, q / 2, 1, 0, q - 1];
        let a = &a_list[..n];
        let m_list = [t - 1, 0, t - 1, 1, 0, 1, t - 1, 0];
        let m = &m_list[..n];
        let other_list = [t - 1, 1, 0, t - 1, 1, 0, 1, t - 1];
        let other = &other_list[..n];
        let mut scratch = vec![93; 3 * n];
        let mut pk = vec![93; 2 * n];
        let mut work = exact_work(Operation::Key, n);
        key_equation_into(params, s, a, e, &mut pk, &mut scratch, &mut work).unwrap();
        assert_eq!(pk, oracle_key(s, a, e, q));
        assert_eq!(work.remaining(), 0);
        assert_eq!(scratch, vec![0; 3 * n]);
        let mut ciphertexts = Vec::new();
        for message in [m, other] {
            scratch.fill(93);
            let mut ct = vec![93; 2 * n];
            let mut work = exact_work(Operation::Encrypt, n);
            encrypt_into(params, &pk, message, u, e0, e1, &mut ct, &mut scratch, &mut work)
                .unwrap();
            assert_eq!(ct, oracle_encrypt(&pk, message, u, e0, e1, q, t));
            assert_eq!(scratch, vec![0; 3 * n]);
            assert_eq!(work.remaining(), 0);
            ciphertexts.push(ct);
        }
        let mut sum = vec![93; 2 * n];
        let mut work = exact_work(Operation::Add, n);
        add_into(params, &ciphertexts[0], &ciphertexts[1], &mut sum, &mut work).unwrap();
        let expected: Vec<u64> = ciphertexts[0]
            .iter()
            .zip(&ciphertexts[1])
            .map(|(&a, &b)| residue(BigInt::from(a) + BigInt::from(b), q))
            .collect();
        assert_eq!(sum, expected);
        assert_eq!(work.remaining(), 0);
        let sum_message: Vec<u64> = m.iter().zip(other).map(|(&a, &b)| (a + b) % t).collect();
        ciphertexts.push(sum);
        for (ct, expected) in ciphertexts.iter().zip([m, other, &sum_message]) {
            scratch.fill(93);
            let mut decoded = vec![93; n];
            let mut work = exact_work(Operation::Decrypt, n);
            decrypt_into(params, s, ct, &mut decoded, &mut scratch, &mut work).unwrap();
            assert_eq!(decoded, oracle_decrypt(s, ct, q, t));
            assert_eq!(decoded, expected);
            assert_eq!(scratch, vec![0; 3 * n]);
            assert_eq!(work.remaining(), 0);
        }
    }
}

#[test]
fn scalar_arithmetic_bigint_boundaries() {
    for q in [3, 17, 257, 4095, (1_u64 << 61) - 1] {
        let values = [0, 1, q / 2, q - 2, q - 1];
        for a in values {
            for b in values {
                assert_eq!(ar::add(a, b, q), residue(BigInt::from(a) + BigInt::from(b), q));
                assert_eq!(ar::sub(a, b, q), residue(BigInt::from(a) - BigInt::from(b), q));
                assert_eq!(ar::mul(a, b, q), residue(BigInt::from(a) * BigInt::from(b), q));
            }
        }
        for value in [-(q as i64) + 1, -1, 0, 1, q as i64 - 1] {
            assert_eq!(ar::signed(value, q), residue(BigInt::from(value), q));
        }
    }
    for value in [i64::MIN, -1, 0, 1, i64::MAX] {
        assert_eq!(ar::magnitude(value), i128::from(value).unsigned_abs() as u64);
    }
    for divisor in [6_u128, 34, 514, 8190, (1_u128 << 62) - 2] {
        for numerator in [0, 1, divisor - 1, divisor, u128::MAX, 1_u128 << 127] {
            assert_eq!(
                BigInt::from(ar::divide(numerator, divisor)),
                BigInt::from(numerator) / BigInt::from(divisor)
            );
        }
    }
}

#[test]
fn explicit_large_support_extrema_are_not_a_sampler() {
    let bounds = Supports {
        s: 256,
        u: 0,
        e: 256,
        e0: 0,
        e1: 0,
    };
    let params = Params::new(2, 257, 2, bounds).unwrap();
    let s = [-256, 256];
    let e = [256, -256];
    let a = [256, 255];
    let mut pk = [99; 4];
    let mut scratch = [99; 6];
    let mut work = exact_work(Operation::Key, 2);
    key_equation_into(params, &s, &a, &e, &mut pk, &mut scratch, &mut work).unwrap();
    assert_eq!(pk.to_vec(), oracle_key(&s, &a, &e, 257));
    assert_eq!(scratch, [0; 6]);
    let mut ct = [99; 4];
    let mut work = exact_work(Operation::Encrypt, 2);
    encrypt_into(
        params,
        &pk,
        &[1, 0],
        &[0, 0],
        &[0, 0],
        &[0, 0],
        &mut ct,
        &mut scratch,
        &mut work,
    )
    .unwrap();
    assert_eq!(ct.to_vec(), oracle_encrypt(&pk, &[1, 0], &[0, 0], &[0, 0], &[0, 0], 257, 2));
    let mut message = [99; 2];
    let mut work = exact_work(Operation::Decrypt, 2);
    decrypt_into(params, &s, &ct, &mut message, &mut scratch, &mut work).unwrap();
    assert_eq!(message, [1, 0]);
    assert_eq!(message.to_vec(), oracle_decrypt(&s, &ct, 257, 2));
    assert_eq!(scratch, [0; 6]);
}

#[test]
fn arbitrary_canonical_rounding_and_cancellation() {
    // Exhaustive small phases plus explicitly selected large boundaries. These
    // are arbitrary raw ciphertexts, not claims about additional valid adds.
    for (q, t) in [(257, 16), (4095, 16)] {
        let params = Params::new(1, q, t, supports(0)).unwrap();
        for phase in 0..q {
            let mut message = [99];
            let mut scratch = [99; 3];
            let mut work = exact_work(Operation::Decrypt, 1);
            decrypt_into(params, &[0], &[phase, q - 1], &mut message, &mut scratch, &mut work)
                .unwrap();
            assert_eq!(message[0], oracle_round(phase, q, t));
            assert_eq!(scratch, [0; 3]);
        }
    }
    let q = (1_u64 << 61) - 1;
    let params = Params::new(1, q, 16, supports(1)).unwrap();
    for phase in [0, 1, q / 32, q / 32 + 1, q / 2, q / 2 + 1, q - 2, q - 1] {
        // c1=q-1, s=1; c0=phase+1 mod q exercises cancellation and wrap.
        let ct = [(phase + 1) % q, q - 1];
        let mut message = [99];
        let mut scratch = [99; 3];
        let mut work = exact_work(Operation::Decrypt, 1);
        decrypt_into(params, &[1], &ct, &mut message, &mut scratch, &mut work).unwrap();
        assert_eq!(message[0], oracle_round(phase, q, 16));
        assert_eq!(message.to_vec(), oracle_decrypt(&[1], &ct, q, 16));
    }
}

#[test]
fn floor_delta_gap_rejection_witness() {
    assert_eq!(Params::new(1, 17, 8, supports(0)), Err(KernelError::Parameters));
    let pk = oracle_key(&[0], &[0], &[0], 17);
    let ct = oracle_encrypt(&pk, &[7], &[0], &[0], &[0], 17, 8);
    let added: Vec<u64> = ct.iter().map(|&v| residue(BigInt::from(v) * 2, 17)).collect();
    assert_eq!(added, [11, 0]);
    assert_eq!(oracle_decrypt(&[0], &added, 17, 8), [5]);
    assert_ne!(5, (7 + 7) % 8);
}

/// Owns only ordinary test vectors; production has no analogous allocation.
struct Buffers {
    s: Vec<i64>,
    a: Vec<u64>,
    e: Vec<i64>,
    pk: Vec<u64>,
    message: Vec<u64>,
    u: Vec<i64>,
    e0: Vec<i64>,
    e1: Vec<i64>,
    lhs: Vec<u64>,
    rhs: Vec<u64>,
    output: Vec<u64>,
    scratch: Vec<u64>,
}

impl Buffers {
    fn new(operation: Operation) -> Self {
        Self {
            s: vec![0; 4],
            a: vec![0; 4],
            e: vec![0; 4],
            pk: vec![0; 8],
            message: vec![0; 4],
            u: vec![0; 4],
            e0: vec![0; 4],
            e1: vec![0; 4],
            lhs: vec![0; 8],
            rhs: vec![0; 8],
            output: vec![99; if operation == Operation::Decrypt { 4 } else { 8 }],
            scratch: vec![99; 12],
        }
    }

    fn call(
        &mut self,
        operation: Operation,
        work: &mut KernelWorkLimit,
    ) -> Result<(), KernelError> {
        let params = Params::new(4, 257, 2, supports(1)).unwrap();
        match operation {
            Operation::Key => key_equation_into(
                params,
                &self.s,
                &self.a,
                &self.e,
                &mut self.output,
                &mut self.scratch,
                work,
            ),
            Operation::Encrypt => encrypt_into(
                params,
                &self.pk,
                &self.message,
                &self.u,
                &self.e0,
                &self.e1,
                &mut self.output,
                &mut self.scratch,
                work,
            ),
            Operation::Add => add_into(params, &self.lhs, &self.rhs, &mut self.output, work),
            Operation::Decrypt => decrypt_into(
                params,
                &self.s,
                &self.lhs,
                &mut self.output,
                &mut self.scratch,
                work,
            ),
        }
    }

    fn resize_slot(&mut self, operation: Operation, slot: usize, extra: bool) {
        // Every actual operation buffer appears once, including output/scratch.
        let delta = if extra { 1 } else { -1 };
        let resize_i64 = |values: &mut Vec<i64>| {
            values.resize((values.len() as isize + delta) as usize, 0);
        };
        let resize_u64 = |values: &mut Vec<u64>| {
            values.resize((values.len() as isize + delta) as usize, 99);
        };
        match (operation, slot) {
            (Operation::Key, 0) | (Operation::Decrypt, 0) => resize_i64(&mut self.s),
            (Operation::Key, 1) => resize_u64(&mut self.a),
            (Operation::Key, 2) => resize_i64(&mut self.e),
            (Operation::Encrypt, 0) => resize_u64(&mut self.pk),
            (Operation::Encrypt, 1) => resize_u64(&mut self.message),
            (Operation::Encrypt, 2) => resize_i64(&mut self.u),
            (Operation::Encrypt, 3) => resize_i64(&mut self.e0),
            (Operation::Encrypt, 4) => resize_i64(&mut self.e1),
            (Operation::Add, 0) | (Operation::Decrypt, 1) => resize_u64(&mut self.lhs),
            (Operation::Add, 1) => resize_u64(&mut self.rhs),
            (Operation::Key, 3)
            | (Operation::Encrypt, 5)
            | (Operation::Add, 2)
            | (Operation::Decrypt, 2) => resize_u64(&mut self.output),
            (Operation::Key, 4) | (Operation::Encrypt, 6) | (Operation::Decrypt, 3) => {
                resize_u64(&mut self.scratch);
            }
            _ => panic!("invalid public test slot"),
        }
    }
}

#[test]
fn every_buffer_geometry_preserves_preflight_state() {
    for (operation, slots) in [
        (Operation::Key, 5),
        (Operation::Encrypt, 7),
        (Operation::Add, 3),
        (Operation::Decrypt, 4),
    ] {
        for slot in 0..slots {
            for extra in [false, true] {
                let mut buffers = Buffers::new(operation);
                buffers.resize_slot(operation, slot, extra);
                let output = buffers.output.clone();
                let scratch = buffers.scratch.clone();
                let mut work = exact_work(operation, 4);
                let remaining = work.remaining();
                assert_eq!(buffers.call(operation, &mut work), Err(KernelError::Geometry));
                assert_eq!(buffers.output, output);
                assert_eq!(buffers.scratch, scratch);
                assert_eq!(work.remaining(), remaining);
            }
        }
    }
}

#[test]
fn late_invalid_values_charged_output_preserved_scratch_wiped() {
    // Each signed slot: positive/negative excess and both i64 extremes.
    // Each canonical slot: modulus and u64::MAX at LAST coefficient.
    for (operation, signed_slots, canonical_slots) in [
        (Operation::Key, 2, 1),
        (Operation::Encrypt, 3, 2),
        (Operation::Add, 0, 2),
        (Operation::Decrypt, 1, 1),
    ] {
        for slot in 0..signed_slots {
            for value in [2, -2, i64::MIN, i64::MAX] {
                let mut buffers = Buffers::new(operation);
                let values = match (operation, slot) {
                    (Operation::Key, 0) | (Operation::Decrypt, 0) => &mut buffers.s,
                    (Operation::Key, 1) => &mut buffers.e,
                    (Operation::Encrypt, 0) => &mut buffers.u,
                    (Operation::Encrypt, 1) => &mut buffers.e0,
                    (Operation::Encrypt, 2) => &mut buffers.e1,
                    _ => panic!("invalid public test slot"),
                };
                *values.last_mut().unwrap() = value;
                assert_value_rejection(&mut buffers, operation);
            }
        }
        for slot in 0..canonical_slots {
            let modulus = if operation == Operation::Encrypt && slot == 1 {
                2
            } else {
                257
            };
            for value in [modulus, u64::MAX] {
                let mut buffers = Buffers::new(operation);
                let values = match (operation, slot) {
                    (Operation::Key, 0) => &mut buffers.a,
                    (Operation::Encrypt, 0) => &mut buffers.pk,
                    (Operation::Encrypt, 1) => &mut buffers.message,
                    (Operation::Add, 0) | (Operation::Decrypt, 0) => &mut buffers.lhs,
                    (Operation::Add, 1) => &mut buffers.rhs,
                    _ => panic!("invalid public test slot"),
                };
                *values.last_mut().unwrap() = value;
                assert_value_rejection(&mut buffers, operation);
            }
        }
    }
}

fn assert_value_rejection(buffers: &mut Buffers, operation: Operation) {
    let before = buffers.output.clone();
    let mut work = exact_work(operation, 4);
    assert_eq!(buffers.call(operation, &mut work), Err(KernelError::InvalidValue));
    assert_eq!(work.remaining(), 0);
    assert_eq!(buffers.output, before);
    assert_eq!(buffers.scratch, vec![if operation == Operation::Add { 99 } else { 0 }; 12]);
}

#[test]
fn exact_and_one_short_work_limits() {
    for (operation, expected_units) in [
        (Operation::Key, 1096),
        (Operation::Encrypt, 2424),
        (Operation::Add, 40),
        (Operation::Decrypt, 1632),
    ] {
        let units = quote(operation, 4).unwrap();
        assert_eq!(units, expected_units);
        let mut buffers = Buffers::new(operation);
        let output = buffers.output.clone();
        let scratch = buffers.scratch.clone();
        let mut short = KernelWorkLimit::new(units - 1);
        assert_eq!(buffers.call(operation, &mut short), Err(KernelError::Work));
        assert_eq!(short.remaining(), units - 1);
        assert_eq!(buffers.output, output);
        assert_eq!(buffers.scratch, scratch);
        let mut exact = KernelWorkLimit::new(units);
        buffers.call(operation, &mut exact).unwrap();
        assert_eq!(exact.remaining(), 0);
        assert_eq!(buffers.scratch, vec![if operation == Operation::Add { 99 } else { 0 }; 12]);
        let after_success = buffers.output.clone();
        buffers.scratch.fill(99);
        assert_eq!(buffers.call(operation, &mut exact), Err(KernelError::Work));
        assert_eq!(buffers.output, after_success);
        assert_eq!(buffers.scratch, vec![99; 12]);
    }
}

#[test]
fn parameter_geometry_and_quote_overflow_rejections() {
    for (n, q, t) in [
        (0, 257, 2),
        (3, 257, 2),
        (8192, 257, 2),
        (1, 2, 2),
        (1, 258, 2),
        (1, 1_u64 << 61, 2),
        (1, 257, 1),
        (1, 257, 257),
    ] {
        assert_eq!(Params::new(n, q, t, supports(0)), Err(KernelError::Parameters));
    }
    for slot in 0..5 {
        let mut bounds = supports(0);
        match slot {
            0 => bounds.s = 257,
            1 => bounds.u = 257,
            2 => bounds.e = 257,
            3 => bounds.e0 = 257,
            4 => bounds.e1 = 257,
            _ => unreachable!(),
        }
        assert_eq!(Params::new(1, 257, 2, bounds), Err(KernelError::Parameters));
    }
    assert!(Params::new(4, 257, 2, supports(1)).is_ok());
    assert!(Params::new(4, 4095, 16, supports(1)).is_ok());
    assert!(Params::new(4096, 257, 2, supports(0)).is_ok());
    let q = (1_u64 << 61) - 1;
    assert_eq!(Params::new(4096, q, 2, supports(q - 1)), Err(KernelError::Overflow));
    assert_eq!(geometry(usize::MAX), Err(KernelError::Overflow));
    assert_eq!(geometry(usize::MAX / 2), Err(KernelError::Overflow));
    for operation in [Operation::Key, Operation::Encrypt, Operation::Decrypt] {
        assert_eq!(quote(operation, usize::MAX), Err(KernelError::Overflow));
    }
    if usize::BITS == 64 {
        assert_eq!(quote(Operation::Add, usize::MAX), Err(KernelError::Overflow));
    } else {
        assert!(quote(Operation::Add, usize::MAX).is_ok());
    }
    // Verify overflow rejection at actual preflight seam without reading data.
    let mut forged = Params::new(4, 257, 2, supports(1)).unwrap();
    forged.n = usize::MAX;
    let mut work = KernelWorkLimit::new(100);
    assert_eq!(preflight(forged, Operation::Key, &[], &mut work), Err(KernelError::Overflow));
    assert_eq!(work.remaining(), 100);
    let mut work = KernelWorkLimit::new(7);
    assert_eq!(work.debit(u64::MAX), Err(WorkExhausted));
    assert_eq!(work.remaining(), 7);
}

#[test]
fn guards_wipe_on_unwind_and_disarm_only_destination() {
    let mut scratch = [91; 12];
    let mut output = [92; 8];
    let result = catch_unwind(AssertUnwindSafe(|| {
        let scratch = WipeGuard::new(&mut scratch);
        scratch.words[11] = 1;
        let output = WipeGuard::new(&mut output);
        output.words[0] = 2;
        panic!("bounded guard unwind test");
    }));
    assert!(result.is_err());
    assert_eq!(scratch, [0; 12]);
    assert_eq!(output, [0; 8]);
    {
        let scratch = WipeGuard::new(&mut scratch);
        scratch.words.fill(3);
        let output = WipeGuard::new(&mut output);
        output.words.fill(4);
        output.finish();
    }
    assert_eq!(scratch, [0; 12]);
    assert_eq!(output, [4; 8]);
}

// Opt-in cases below leave every original test body above unchanged.
use super::bfv::{decrypt_ntt_into, encrypt_ntt_into, key_equation_ntt_into};
use super::convolution::{Convolution, NttError, preflight_ntt, product_count, shell_quote};
use crate::coefficient_layout::DegreeBound;
use crate::coefficient_ntt::{FixedError, FixedOperation, NegacyclicNtt, fixed_quote};

fn ntt_plan(n: usize, q: u64, psi: u64) -> NegacyclicNtt {
    NegacyclicNtt::new(CoefficientRing::new(n, q).unwrap(), psi).unwrap()
}

fn ntt_limits(
    operation: Operation,
    n: usize,
    plan: &NegacyclicNtt,
) -> (KernelWorkLimit, KernelWorkLimit) {
    let shell = shell_quote(operation, n).unwrap();
    let products = product_count(operation).unwrap();
    let ntt = fixed_quote(plan, FixedOperation::Product)
        .unwrap()
        .total()
        .checked_mul(products)
        .unwrap();
    (KernelWorkLimit::new(shell), KernelWorkLimit::new(ntt))
}

fn call_ntt_buffers(
    buffers: &mut Buffers,
    params: Params,
    operation: Operation,
    plan: &NegacyclicNtt,
    workspace: &mut [u64],
    shell: &mut KernelWorkLimit,
    ntt: &mut KernelWorkLimit,
) -> Result<(), KernelError> {
    match operation {
        Operation::Key => key_equation_ntt_into(
            params,
            &buffers.s,
            &buffers.a,
            &buffers.e,
            &mut buffers.output,
            &mut buffers.scratch,
            shell,
            plan,
            workspace,
            ntt,
        ),
        Operation::Encrypt => encrypt_ntt_into(
            params,
            &buffers.pk,
            &buffers.message,
            &buffers.u,
            &buffers.e0,
            &buffers.e1,
            &mut buffers.output,
            &mut buffers.scratch,
            shell,
            plan,
            workspace,
            ntt,
        ),
        Operation::Decrypt => decrypt_ntt_into(
            params,
            &buffers.s,
            &buffers.lhs,
            &mut buffers.output,
            &mut buffers.scratch,
            shell,
            plan,
            workspace,
            ntt,
        ),
        Operation::Add => panic!("no opt-in addition dispatch"),
    }
}

#[test]
fn ntt_equations_one_add_bigint_and_schoolbook_parity() {
    let large = (1_u64 << 61) - 1;
    for (n, q, t, psi) in [
        (1, 257, 2, 256),
        (2, 257, 2, 16),
        (4, 257, 2, 4),
        (8, 257, 2, 2),
        (1, large, 16, large - 1),
    ] {
        let params = Params::new(n, q, t, supports(1)).unwrap();
        let plan = ntt_plan(n, q, psi);
        let secret = [-1, 1, 0, -1, 1, 0, 1, -1];
        let ephemeral = [1, -1, 1, 0, -1, 1, 0, -1];
        let error = [-1, 1, -1, 1, 0, -1, 1, 0];
        let error0 = [1, -1, 0, 1, -1, 0, 1, -1];
        let error1 = [-1, 0, 1, -1, 1, -1, 0, 1];
        let public = [q - 1, 0, 1, q - 2, q / 2, 1, 0, q - 1];
        let first_message = [t - 1, 0, t - 1, 1, 0, 1, t - 1, 0];
        let second_message = [t - 1, 1, 0, t - 1, 1, 0, 1, t - 1];
        let s = &secret[..n];
        let u = &ephemeral[..n];
        let e = &error[..n];
        let e0 = &error0[..n];
        let e1 = &error1[..n];
        let a = &public[..n];
        let mut scratch = vec![91; 3 * n];
        let mut workspace = vec![92; 2 * n];
        let mut reference_scratch = vec![93; 3 * n];
        let mut pk = vec![99; 2 * n];
        let mut reference_pk = vec![99; 2 * n];
        let (mut shell, mut ntt) = ntt_limits(Operation::Key, n, &plan);
        key_equation_ntt_into(
            params, s, a, e, &mut pk, &mut scratch, &mut shell, &plan, &mut workspace, &mut ntt,
        )
        .unwrap();
        key_equation_into(
            params,
            s,
            a,
            e,
            &mut reference_pk,
            &mut reference_scratch,
            &mut exact_work(Operation::Key, n),
        )
        .unwrap();
        assert_eq!(pk, oracle_key(s, a, e, q));
        assert_eq!(pk, reference_pk);
        assert_eq!(scratch, vec![0; 3 * n]);
        assert_eq!(workspace, vec![0; 2 * n]);
        assert_eq!((shell.remaining(), ntt.remaining()), (0, 0));
        let mut ciphertexts = Vec::new();
        for message in [&first_message[..n], &second_message[..n]] {
            scratch.fill(91);
            workspace.fill(92);
            let mut ct = vec![99; 2 * n];
            let mut reference = vec![99; 2 * n];
            let (mut shell, mut ntt) = ntt_limits(Operation::Encrypt, n, &plan);
            encrypt_ntt_into(
                params,
                &pk,
                message,
                u,
                e0,
                e1,
                &mut ct,
                &mut scratch,
                &mut shell,
                &plan,
                &mut workspace,
                &mut ntt,
            )
            .unwrap();
            encrypt_into(
                params,
                &pk,
                message,
                u,
                e0,
                e1,
                &mut reference,
                &mut reference_scratch,
                &mut exact_work(Operation::Encrypt, n),
            )
            .unwrap();
            assert_eq!(ct, oracle_encrypt(&pk, message, u, e0, e1, q, t));
            assert_eq!(ct, reference);
            assert_eq!(scratch, vec![0; 3 * n]);
            assert_eq!(workspace, vec![0; 2 * n]);
            assert_eq!((shell.remaining(), ntt.remaining()), (0, 0));
            ciphertexts.push(ct);
        }
        let mut sum = vec![99; 2 * n];
        add_into(
            params,
            &ciphertexts[0],
            &ciphertexts[1],
            &mut sum,
            &mut exact_work(Operation::Add, n),
        )
        .unwrap();
        let expected_sum: Vec<u64> = ciphertexts[0]
            .iter()
            .zip(&ciphertexts[1])
            .map(|(&a, &b)| residue(BigInt::from(a) + BigInt::from(b), q))
            .collect();
        assert_eq!(sum, expected_sum);
        ciphertexts.push(sum);
        let sum_message: Vec<u64> = first_message[..n]
            .iter()
            .zip(&second_message[..n])
            .map(|(&a, &b)| residue(BigInt::from(a) + BigInt::from(b), t))
            .collect();
        for (ct, expected) in ciphertexts
            .iter()
            .zip([&first_message[..n], &second_message[..n], &sum_message])
        {
            scratch.fill(91);
            workspace.fill(92);
            let mut message = vec![99; n];
            let mut reference = vec![99; n];
            let (mut shell, mut ntt) = ntt_limits(Operation::Decrypt, n, &plan);
            decrypt_ntt_into(
                params,
                s,
                ct,
                &mut message,
                &mut scratch,
                &mut shell,
                &plan,
                &mut workspace,
                &mut ntt,
            )
            .unwrap();
            decrypt_into(
                params,
                s,
                ct,
                &mut reference,
                &mut reference_scratch,
                &mut exact_work(Operation::Decrypt, n),
            )
            .unwrap();
            assert_eq!(message, oracle_decrypt(s, ct, q, t));
            assert_eq!(message, reference);
            assert_eq!(message, expected);
            assert_eq!(scratch, vec![0; 3 * n]);
            assert_eq!(workspace, vec![0; 2 * n]);
            assert_eq!((shell.remaining(), ntt.remaining()), (0, 0));
        }
    }
}

#[test]
fn ntt_signed_extrema_and_arbitrary_canonical_decryption() {
    let plan = ntt_plan(2, 257, 16);
    let bounds = Supports { s: 256, u: 0, e: 256, e0: 0, e1: 0 };
    let params = Params::new(2, 257, 2, bounds).unwrap();
    let s = [-256, 256];
    let a = [256, 255];
    let e = [256, -256];
    let mut pk = [99; 4];
    let mut scratch = [91; 6];
    let mut workspace = [92; 4];
    let (mut shell, mut ntt) = ntt_limits(Operation::Key, 2, &plan);
    key_equation_ntt_into(
        params, &s, &a, &e, &mut pk, &mut scratch, &mut shell, &plan, &mut workspace, &mut ntt,
    )
    .unwrap();
    assert_eq!(pk.to_vec(), oracle_key(&s, &a, &e, 257));
    assert_eq!(scratch, [0; 6]);
    assert_eq!(workspace, [0; 4]);
    let arbitrary_ct = [256, 255, 254, 256];
    let mut message = [99; 2];
    let (mut shell, mut ntt) = ntt_limits(Operation::Decrypt, 2, &plan);
    decrypt_ntt_into(
        params,
        &s,
        &arbitrary_ct,
        &mut message,
        &mut scratch,
        &mut shell,
        &plan,
        &mut workspace,
        &mut ntt,
    )
    .unwrap();
    assert_eq!(message.to_vec(), oracle_decrypt(&s, &arbitrary_ct, 257, 2));
    assert_eq!(scratch, [0; 6]);
    assert_eq!(workspace, [0; 4]);
    let bounds = Supports { s: 0, u: 256, e: 0, e0: 0, e1: 0 };
    let params = Params::new(2, 257, 2, bounds).unwrap();
    // Arbitrary supplied pk: equation test, not an issued or valid key claim.
    let pk = [256, 255, 254, 256];
    let mut ct = [99; 4];
    let (mut shell, mut ntt) = ntt_limits(Operation::Encrypt, 2, &plan);
    encrypt_ntt_into(
        params,
        &pk,
        &[1, 0],
        &[-256, 256],
        &[0, 0],
        &[0, 0],
        &mut ct,
        &mut scratch,
        &mut shell,
        &plan,
        &mut workspace,
        &mut ntt,
    )
    .unwrap();
    assert_eq!(
        ct.to_vec(),
        oracle_encrypt(&pk, &[1, 0], &[-256, 256], &[0; 2], &[0; 2], 257, 2)
    );
    assert_eq!(scratch, [0; 6]);
    assert_eq!(workspace, [0; 4]);
    for q in [257, (1_u64 << 61) - 1] {
        let params = Params::new(1, q, 16, supports(0)).unwrap();
        let plan = ntt_plan(1, q, q - 1);
        for phase in [0, 1, q / 32, q / 32 + 1, q / 2, q / 2 + 1, q - 2, q - 1] {
            let ct = [phase, q - 1];
            let mut message = [99];
            let mut scratch = [91; 3];
            let mut workspace = [92; 2];
            let (mut shell, mut ntt) = ntt_limits(Operation::Decrypt, 1, &plan);
            decrypt_ntt_into(
                params,
                &[0],
                &ct,
                &mut message,
                &mut scratch,
                &mut shell,
                &plan,
                &mut workspace,
                &mut ntt,
            )
            .unwrap();
            assert_eq!(message.to_vec(), oracle_decrypt(&[0], &ct, q, 16));
            assert_eq!(message[0], oracle_round(phase, q, 16));
            assert_eq!(scratch, [0; 3]);
            assert_eq!(workspace, [0; 2]);
            assert_eq!((shell.remaining(), ntt.remaining()), (0, 0));
        }
    }
}

fn assert_ntt_preflight_refusal(
    params: Params,
    operation: Operation,
    plan: &NegacyclicNtt,
    expected: KernelError,
) {
    let mut buffers = Buffers::new(operation);
    let mut workspace = [92; 8];
    let mut shell = KernelWorkLimit::new(10000);
    let mut ntt = KernelWorkLimit::new(10000);
    let output = buffers.output.clone();
    let scratch = buffers.scratch.clone();
    assert_eq!(
        call_ntt_buffers(
            &mut buffers,
            params,
            operation,
            plan,
            &mut workspace,
            &mut shell,
            &mut ntt,
        ),
        Err(expected)
    );
    assert_eq!(buffers.output, output);
    assert_eq!(buffers.scratch, scratch);
    assert_eq!(workspace, [92; 8]);
    assert_eq!((shell.remaining(), ntt.remaining()), (10000, 10000));
}

#[test]
fn ntt_plan_degree_modulus_and_geometry_preflight_atomic() {
    let params = Params::new(4, 257, 2, supports(1)).unwrap();
    let wrong_degree = ntt_plan(2, 257, 16);
    let wrong_modulus = ntt_plan(4, 17, 2);
    let matching = ntt_plan(4, 257, 4);
    let ring = CoefficientRing::bounded(8192, 65537, DegreeBound::Public8192).unwrap();
    let public_8192 = NegacyclicNtt::new(ring, 81).unwrap();
    let legacy_params = Params::new(4096, 65537, 2, supports(0)).unwrap();
    let legacy_plan = ntt_plan(4096, 65537, 6561);
    for operation in [Operation::Key, Operation::Encrypt, Operation::Decrypt] {
        for plan in [&wrong_degree, &wrong_modulus] {
            assert_ntt_preflight_refusal(
                params, operation, plan, KernelError::Ntt(NttError::PlanMismatch),
            );
        }
        assert_ntt_preflight_refusal(
            params,
            operation,
            &public_8192,
            KernelError::Ntt(NttError::Fixed(FixedError::Degree)),
        );
        // Matching actual degree 4096 reaches shape checks, but this does NOT
        // execute a large transform or schoolbook product in these definitions.
        assert_ntt_preflight_refusal(
            legacy_params, operation, &legacy_plan, KernelError::Geometry,
        );
        let mut inconsistent = params;
        inconsistent.two_n = 7;
        assert_ntt_preflight_refusal(
            inconsistent, operation, &matching, KernelError::Geometry,
        );
        let mut overflow = params;
        overflow.n = usize::MAX;
        assert_ntt_preflight_refusal(overflow, operation, &matching, KernelError::Overflow);
        // Test the independent actual-Params degree gate at the private seam;
        // no unchecked plan or root is forged, and no coefficients are read.
        let mut too_large = legacy_params;
        too_large.n = 8192;
        too_large.two_n = 16384;
        too_large.three_n = 24576;
        assert_ntt_preflight_refusal(
            too_large,
            operation,
            &public_8192,
            KernelError::Ntt(NttError::Fixed(FixedError::Degree)),
        );
    }
}

fn clear_ntt_slot(buffers: &mut Buffers, operation: Operation, slot: usize) {
    match (operation, slot) {
        (Operation::Key, 0) | (Operation::Decrypt, 0) => buffers.s.clear(),
        (Operation::Key, 1) => buffers.a.clear(),
        (Operation::Key, 2) => buffers.e.clear(),
        (Operation::Encrypt, 0) => buffers.pk.clear(),
        (Operation::Encrypt, 1) => buffers.message.clear(),
        (Operation::Encrypt, 2) => buffers.u.clear(),
        (Operation::Encrypt, 3) => buffers.e0.clear(),
        (Operation::Encrypt, 4) => buffers.e1.clear(),
        (Operation::Decrypt, 1) => buffers.lhs.clear(),
        (Operation::Key, 3) | (Operation::Encrypt, 5) | (Operation::Decrypt, 2) => {
            buffers.output.clear();
        }
        (Operation::Key, 4) | (Operation::Encrypt, 6) | (Operation::Decrypt, 3) => {
            buffers.scratch.clear();
        }
        _ => panic!("invalid public opt-in shape slot"),
    }
}

#[test]
fn ntt_all_buffer_shapes_refuse_before_both_debits() {
    let params = Params::new(4, 257, 2, supports(1)).unwrap();
    let plan = ntt_plan(4, 257, 4);
    for (operation, slots) in [
        (Operation::Key, 5),
        (Operation::Encrypt, 7),
        (Operation::Decrypt, 4),
    ] {
        // Final slot is the additional NTT workspace; all original slots retain
        // their existing geometry and error variants, including both outputs.
        for slot in 0..=slots {
            for shape in [0, 1, 2] {
                let mut buffers = Buffers::new(operation);
                let mut workspace = vec![92; 8];
                let expected = if slot == slots {
                    workspace.resize([0, 7, 9][shape], 92);
                    KernelError::Ntt(NttError::Fixed(FixedError::Geometry))
                } else {
                    if shape == 0 {
                        clear_ntt_slot(&mut buffers, operation, slot);
                    } else {
                        buffers.resize_slot(operation, slot, shape == 2);
                    }
                    KernelError::Geometry
                };
                let output = buffers.output.clone();
                let scratch = buffers.scratch.clone();
                let before_workspace = workspace.clone();
                let (mut shell, mut ntt) = ntt_limits(operation, 4, &plan);
                let before = (shell.remaining(), ntt.remaining());
                assert_eq!(
                    call_ntt_buffers(
                        &mut buffers,
                        params,
                        operation,
                        &plan,
                        &mut workspace,
                        &mut shell,
                        &mut ntt,
                    ),
                    Err(expected)
                );
                assert_eq!((shell.remaining(), ntt.remaining()), before);
                assert_eq!(buffers.output, output);
                assert_eq!(buffers.scratch, scratch);
                assert_eq!(workspace, before_workspace);
            }
        }
    }
}

#[test]
fn ntt_both_balances_preflight_exact_and_cumulative() {
    let params = Params::new(4, 257, 2, supports(1)).unwrap();
    let plan = ntt_plan(4, 257, 4);
    assert_eq!(fixed_quote(&plan, FixedOperation::Product).unwrap().total(), 227);
    for (operation, shell_units, ntt_units) in [
        (Operation::Key, 64, 227),
        (Operation::Encrypt, 352, 454),
        (Operation::Decrypt, 600, 227),
    ] {
        for (shell_balance, ntt_balance, expected) in [
            (shell_units - 1, ntt_units, KernelError::Work),
            (shell_units, ntt_units - 1, KernelError::Ntt(NttError::Fixed(FixedError::Work))),
            (shell_units - 1, ntt_units - 1, KernelError::Work),
        ] {
            let mut buffers = Buffers::new(operation);
            let mut workspace = [92; 8];
            let mut shell = KernelWorkLimit::new(shell_balance);
            let mut ntt = KernelWorkLimit::new(ntt_balance);
            let output = buffers.output.clone();
            assert_eq!(
                call_ntt_buffers(
                    &mut buffers,
                    params,
                    operation,
                    &plan,
                    &mut workspace,
                    &mut shell,
                    &mut ntt,
                ),
                Err(expected)
            );
            assert_eq!((shell.remaining(), ntt.remaining()), (shell_balance, ntt_balance));
            assert_eq!(buffers.output, output);
            assert_eq!(buffers.scratch, vec![99; 12]);
            assert_eq!(workspace, [92; 8]);
        }
        // For encryption, 453 units could execute ONE product. The refusal
        // above must nevertheless precede the first product and BOTH debits.
        let mut buffers = Buffers::new(operation);
        let mut workspace = [92; 8];
        let mut shell = KernelWorkLimit::new(2 * shell_units);
        let mut ntt = KernelWorkLimit::new(2 * ntt_units);
        for remaining_calls in [1, 0] {
            call_ntt_buffers(
                &mut buffers,
                params,
                operation,
                &plan,
                &mut workspace,
                &mut shell,
                &mut ntt,
            )
            .unwrap();
            assert_eq!(
                (shell.remaining(), ntt.remaining()),
                (remaining_calls * shell_units, remaining_calls * ntt_units)
            );
            assert_eq!(buffers.output, vec![0; buffers.output.len()]);
            assert_eq!(buffers.scratch, vec![0; 12]);
            assert_eq!(workspace, [0; 8]);
            buffers.scratch.fill(99);
            workspace.fill(92);
        }
        let output = buffers.output.clone();
        assert_eq!(
            call_ntt_buffers(
                &mut buffers,
                params,
                operation,
                &plan,
                &mut workspace,
                &mut shell,
                &mut ntt,
            ),
            Err(KernelError::Work)
        );
        assert_eq!(buffers.output, output);
        assert_eq!(buffers.scratch, vec![99; 12]);
        assert_eq!(workspace, [92; 8]);
        assert_eq!((shell.remaining(), ntt.remaining()), (0, 0));
    }
}

fn assert_ntt_value_refusal(buffers: &mut Buffers, operation: Operation) {
    let params = Params::new(4, 257, 2, supports(1)).unwrap();
    let plan = ntt_plan(4, 257, 4);
    let (mut shell, mut ntt) = ntt_limits(operation, 4, &plan);
    let ntt_before = ntt.remaining();
    let output = buffers.output.clone();
    let mut workspace = [92; 8];
    assert_eq!(
        call_ntt_buffers(buffers, params, operation, &plan, &mut workspace, &mut shell, &mut ntt),
        Err(KernelError::InvalidValue)
    );
    assert_eq!(shell.remaining(), 0);
    assert_eq!(ntt.remaining(), ntt_before);
    assert_eq!(buffers.output, output);
    assert_eq!(buffers.scratch, vec![0; 12]);
    assert_eq!(workspace, [0; 8]);
}

#[test]
fn ntt_first_and_late_invalid_values_spend_only_shell() {
    for (operation, signed_slots, canonical_slots) in [
        (Operation::Key, 2, 1),
        (Operation::Encrypt, 3, 2),
        (Operation::Decrypt, 1, 1),
    ] {
        for slot in 0..signed_slots {
            for last in [false, true] {
                for value in [2, -2, i64::MIN, i64::MAX] {
                    let mut buffers = Buffers::new(operation);
                    let values = match (operation, slot) {
                        (Operation::Key, 0) | (Operation::Decrypt, 0) => &mut buffers.s,
                        (Operation::Key, 1) => &mut buffers.e,
                        (Operation::Encrypt, 0) => &mut buffers.u,
                        (Operation::Encrypt, 1) => &mut buffers.e0,
                        (Operation::Encrypt, 2) => &mut buffers.e1,
                        _ => panic!("invalid public signed opt-in slot"),
                    };
                    let at = if last { values.len() - 1 } else { 0 };
                    values[at] = value;
                    assert_ntt_value_refusal(&mut buffers, operation);
                }
            }
        }
        for slot in 0..canonical_slots {
            let modulus = if operation == Operation::Encrypt && slot == 1 { 2 } else { 257 };
            for last in [false, true] {
                for value in [modulus, u64::MAX] {
                    let mut buffers = Buffers::new(operation);
                    let values = match (operation, slot) {
                        (Operation::Key, 0) => &mut buffers.a,
                        (Operation::Encrypt, 0) => &mut buffers.pk,
                        (Operation::Encrypt, 1) => &mut buffers.message,
                        (Operation::Decrypt, 0) => &mut buffers.lhs,
                        _ => panic!("invalid public canonical opt-in slot"),
                    };
                    let at = if last { values.len() - 1 } else { 0 };
                    values[at] = value;
                    assert_ntt_value_refusal(&mut buffers, operation);
                }
            }
        }
    }
}

#[test]
fn ntt_closed_dispatch_failures_and_outer_guard_unwind() {
    let params = Params::new(4, 257, 2, supports(1)).unwrap();
    let plan = ntt_plan(4, 257, 4);
    let units = fixed_quote(&plan, FixedOperation::Product).unwrap().total();
    let a = [256, 0, 1, 255];
    let secret_words = [256, 1, 0, 256];
    let mut scratch = [91; 12];
    let mut workspace = [92; 8];
    let mut ntt = KernelWorkLimit::new(units);
    {
        // Exercise the real closed-dispatch seam, deliberately WITHOUT the
        // equation wrapper's two-product budget precheck. A later work refusal
        // is unreachable with a correctly prechecked wrapper and exclusive
        // counter borrow. This does not inject failures into production code.
        let scratch = WipeGuard::new(&mut scratch);
        let workspace = WipeGuard::new(&mut workspace);
        let (secret, rest) = scratch.words.split_at_mut(4);
        secret.copy_from_slice(&secret_words);
        let (first, second) = rest.split_at_mut(4);
        let mut dispatch = Convolution::FixedNtt {
            plan: &plan,
            scratch: &mut *workspace.words,
            work: &mut ntt,
        };
        dispatch.product(&a, secret, first, 257).unwrap();
        let expected: Vec<u64> = whole_convolution(&unsigned_words(&a), &unsigned_words(secret))
            .into_iter()
            .map(|value| residue(value, 257))
            .collect();
        assert_eq!(&*first, expected.as_slice());
        // Model a secret-bearing gap to test the OUTER guard, not the nested
        // successful NTT wipe. No such mutation hook exists in production.
        if let Convolution::FixedNtt { scratch, .. } = &mut dispatch {
            scratch.fill(87);
        }
        assert_eq!(
            dispatch.product(&a, secret, second, 257),
            Err(KernelError::Ntt(NttError::Fixed(FixedError::Work)))
        );
        assert_eq!(&*second, &[91; 4]);
        if let Convolution::FixedNtt { scratch, .. } = &dispatch {
            assert_eq!(&**scratch, &[87; 8]);
        }
    }
    assert_eq!(ntt.remaining(), 0);
    assert_eq!(scratch, [0; 12]);
    assert_eq!(workspace, [0; 8]);

    let mut output = [99; 8];
    let mut shell = KernelWorkLimit::new(64);
    let mut ntt = KernelWorkLimit::new(units);
    preflight_ntt(
        params,
        Operation::Key,
        &[
            (secret_words.len(), 4),
            (a.len(), 4),
            (4, 4),
            (output.len(), 8),
            (scratch.len(), 12),
        ],
        &mut shell,
        &plan,
        workspace.len(),
        &mut ntt,
    )
    .unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let workspace = WipeGuard::new(&mut workspace);
        let scratch = WipeGuard::new(&mut scratch);
        let (secret, rest) = scratch.words.split_at_mut(4);
        secret.copy_from_slice(&secret_words);
        let (product, _) = rest.split_at_mut(4);
        let mut dispatch = Convolution::FixedNtt {
            plan: &plan,
            scratch: &mut *workspace.words,
            work: &mut ntt,
        };
        dispatch.product(&a, secret, product, 257).unwrap();
        if let Convolution::FixedNtt { scratch, .. } = &mut dispatch {
            scratch.fill(88);
        }
        let output = WipeGuard::new(&mut output);
        output.words[0] = 1;
        panic!("bounded opt-in shared-guard unwind");
    }));
    assert!(result.is_err());
    assert_eq!(scratch, [0; 12]);
    assert_eq!(workspace, [0; 8]);
    assert_eq!(output, [0; 8]);
    assert_eq!((shell.remaining(), ntt.remaining()), (0, 0));
    // These are guard/dispatch mechanism tests, NOT injected panics in every
    // production equation or an abort/register-copy/erasure certification.
}

#[test]
fn ntt_default_composite_modulus_and_quotes_unchanged() {
    let params = Params::new(4, 4095, 16, supports(1)).unwrap();
    let s = [-1, 1, 0, 1];
    let a = [4094, 0, 1, 4093];
    let e = [1, -1, 0, -1];
    let u = [1, -1, 0, 1];
    let e0 = [-1, 0, 1, -1];
    let e1 = [0, 1, -1, 1];
    let message = [15, 0, 1, 15];
    let mut scratch = [91; 12];
    let mut pk = [99; 8];
    let mut ct = [99; 8];
    let mut decoded = [99; 4];
    for (operation, expected) in [
        (Operation::Key, 1096),
        (Operation::Encrypt, 2424),
        (Operation::Decrypt, 1632),
        (Operation::Add, 40),
    ] {
        assert_eq!(quote(operation, 4).unwrap(), expected);
    }
    key_equation_into(
        params, &s, &a, &e, &mut pk, &mut scratch, &mut exact_work(Operation::Key, 4),
    )
    .unwrap();
    assert_eq!(pk.to_vec(), oracle_key(&s, &a, &e, 4095));
    encrypt_into(
        params,
        &pk,
        &message,
        &u,
        &e0,
        &e1,
        &mut ct,
        &mut scratch,
        &mut exact_work(Operation::Encrypt, 4),
    )
    .unwrap();
    assert_eq!(ct.to_vec(), oracle_encrypt(&pk, &message, &u, &e0, &e1, 4095, 16));
    decrypt_into(
        params, &s, &ct, &mut decoded, &mut scratch, &mut exact_work(Operation::Decrypt, 4),
    )
    .unwrap();
    assert_eq!(decoded, message);
    assert_eq!(decoded.to_vec(), oracle_decrypt(&s, &ct, 4095, 16));
    assert_eq!(scratch, [0; 12]);
    let plan = ntt_plan(4, 257, 4);
    assert_ntt_preflight_refusal(
        params, Operation::Key, &plan, KernelError::Ntt(NttError::PlanMismatch),
    );
}

#[test]
fn ntt_shell_quote_and_preflight_overflow_rejections() {
    for (operation, weight, count) in [
        (Operation::Key, 16, 1),
        (Operation::Encrypt, 88, 2),
        (Operation::Decrypt, 150, 1),
    ] {
        assert_eq!(product_count(operation).unwrap(), count);
        assert_eq!(shell_quote(operation, 1).unwrap(), weight);
        assert_eq!(shell_quote(operation, 4096).unwrap(), weight * 4096);
        if usize::BITS == 64 {
            assert_eq!(shell_quote(operation, usize::MAX), Err(KernelError::Overflow));
        } else {
            assert!(shell_quote(operation, usize::MAX).is_ok());
        }
    }
    assert_eq!(shell_quote(Operation::Add, 4), Err(KernelError::Parameters));
    assert_eq!(product_count(Operation::Add), Err(KernelError::Parameters));
}
