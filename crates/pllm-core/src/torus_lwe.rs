//! NONSECURE TOY: noisy same-width Torus64 identity key switching.
//!
//! Independently authored equations; HE3DB commit
//! 1d8d7827baf32b4094b7547aeeaa9faf59253b26, keyswitch.cpp IdentityKeySwitch
//! is provenance only. Its 64->32 conversion is NOT reproduced here.
//! Binary independent 8/4 secrets, radix 16, eight HIGH digits, phase b-a.s.
//! Errors are independent uniform {-1,+1}, NOT Gaussian or paper parameters.
//! Reusing the evaluation key correlates output errors; this is standard HE
//! key reuse, not consumable Beaver material. No production/security adoption.
//!
//! Private owners are neither Clone nor Debug; no serialization. Ordinary Drop
//! and unwind wipe owned buffers. Aborts, registers, compiler/saved copies and
//! allocator/RSS guarantees are excluded. The fixed facade exports aggregates
//! only, not general cryptographic operations or a production executor.

use crate::kernel_support::{KernelWorkLimit, WipeGuard};
use std::{cell::Cell, sync::Arc, time::Instant};
use zeroize::{Zeroize, Zeroizing};

pub const BASE_LOG: usize = 4;
pub const ROWS: usize = 8 * 8 * 15;
pub const TABLE_BYTES: usize = ROWS * 5 * 8;
pub const REQUESTED_OS_BYTES: usize = 12 + ROWS * 33 + 4 * 65;
pub const PAYLOAD_BYTES: usize = 4 * (9 + 5) * 8;
pub const MAX_LOGICAL_BYTES: usize = 64 << 20;
pub const ERROR_BOUND: u64 = 8 * (1u64 << 31) + 65;
pub const CONTRACT: &str = "toy-torus64-lwe-8-to-4-radix16-high8-halfup-pm1-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error { Overflow, Budget, Entropy, Binding, Terminal, Verification }

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "torus LWE toy: {self:?}") }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Quote {
    pub logical_bytes: usize,
    pub work_units: u64,
    pub requested_os_bytes: usize,
}

// Conservative named-event inventory, not machine instructions or CPU fuel.
// Includes encryption products, all possible selected word subtractions,
// oracle quotient/digit/products, initialization, entropy parsing, message
// generation, rounding, counters and retirement wipes, even on zero digits.
fn inventory(repetitions: usize) -> Result<Quote, Error> {
    let material = TABLE_BYTES.checked_add(ROWS).and_then(|n| n.checked_add(12))
        .and_then(|n| n.checked_add(PAYLOAD_BYTES))
        .and_then(|n| n.checked_add(REQUESTED_OS_BYTES)).ok_or(Error::Overflow)?;
    // Stack row/digits/oracle scratch 4096; Arc allocation/control, owned
    // descriptors, alignment/partial initialization 4096; bounded native/Python
    // report/JSON copies 131072. Fingerprint workspace priced separately by binding.
    let bytes = material.checked_add(4096 + 4096 + 131072).ok_or(Error::Overflow)?;
    let work = 3 * REQUESTED_OS_BYTES as u64 // entropy byte init/fill/wipe
        + 4 * 12 // secret sample/init/two retirement wipes
        + ROWS as u64 * (5 + 5 + 10) // table init/write/two wipes, word events
        + ROWS as u64 * (32 + 4 + 4 + 5 + 1 + 3 + 5 + 10) // temporary row parse/masks/products/adds/error/message/init/two wipes
        + (ROWS as u64 + 4) * 4 // oracle ledger init/write/two wipes
        + 4 * (64 + 8 + 8 + 1 + 9 + 9 + 18 + 1) // input byte parse/products/adds/error/write/init/two wipes/message
        + 4 * (5 + 1 + 10 + 8) // output init/copy/two wipes/rounding
        + 256 * (1 + 5) // digit extraction and maximum row subtractions
        + 4 * (4 + 8 * 8 + 64 * 8 + 16) // decrypt and independent quotient/digit oracle
        + 4096; // state, counters, public facade/check and scalar scratch overhead
    Ok(Quote { logical_bytes: bytes.checked_mul(repetitions).ok_or(Error::Overflow)?,
        work_units: work.checked_mul(u64::try_from(repetitions).map_err(|_| Error::Overflow)?).ok_or(Error::Overflow)?,
        requested_os_bytes: REQUESTED_OS_BYTES.checked_mul(repetitions).ok_or(Error::Overflow)? })
}

/// Deterministic, allocation/entropy-free admission. Ceiling is logical estimate,
/// NOT a reservation, enforced RSS limit, or proof the entire wrapper fits.
pub fn preflight(max_bytes: usize, max_work: u64) -> Result<Quote, Error> {
    let q = inventory(1)?;
    if q.logical_bytes > max_bytes.min(MAX_LOGICAL_BYTES) || q.work_units > max_work { return Err(Error::Budget); }
    Ok(q)
}

struct Anchor { _live: u8 }
struct InputClient { id: Arc<Anchor>, secret: Zeroizing<Vec<u8>> }
struct OutputClient { id: Arc<Anchor>, secret: Zeroizing<Vec<u8>>, terminal: Cell<bool> }
struct Evaluator { id: Arc<Anchor>, table: Zeroizing<Vec<u64>>, terminal: bool }
struct Oracle { id: Arc<Anchor>, errors: Zeroizing<Vec<i8>>, input_errors: Zeroizing<Vec<i8>>, terminal: Cell<bool> }
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role { Input, Output }
struct Ciphertext { id: Arc<Anchor>, role: Role, words: Zeroizing<Vec<u64>> }
struct Session { input: InputClient, output: OutputClient, evaluator: Evaluator, oracle: Oracle,
    ciphertexts: Vec<Ciphertext>, terminal: bool }

// Explicit retirement also makes the real owned-owner Drop path observable by
// unit tests before deallocation. Zeroizing remains the persistent backstop.
impl Drop for InputClient {
    fn drop(&mut self) {
        self.secret.as_mut_slice().zeroize();
        #[cfg(test)] tests::observe_wipe(0, self.secret.len(), self.secret.iter().all(|&x| x == 0));
    }
}
impl Drop for OutputClient {
    fn drop(&mut self) {
        self.terminal.set(true);
        self.secret.as_mut_slice().zeroize();
        #[cfg(test)] tests::observe_wipe(1, self.secret.len(), self.secret.iter().all(|&x| x == 0));
    }
}
impl Drop for Evaluator {
    fn drop(&mut self) {
        self.terminal = true;
        self.table.as_mut_slice().zeroize();
        #[cfg(test)] tests::observe_wipe(2, self.table.len(), self.table.iter().all(|&x| x == 0));
    }
}
impl Drop for Oracle {
    fn drop(&mut self) {
        self.terminal.set(true);
        self.errors.as_mut_slice().zeroize();
        self.input_errors.as_mut_slice().zeroize();
        #[cfg(test)] tests::observe_wipe(3, self.errors.len() + self.input_errors.len(),
            self.errors.iter().chain(self.input_errors.iter()).all(|&x| x == 0));
    }
}
impl Drop for Ciphertext {
    fn drop(&mut self) {
        self.words.as_mut_slice().zeroize();
        #[cfg(test)] tests::observe_wipe(4, self.words.len(), self.words.iter().all(|&x| x == 0));
    }
}
impl Drop for Session {
    fn drop(&mut self) { self.terminal = true; }
}

fn sign(byte: u8) -> i8 { if byte & 1 == 0 { -1 } else { 1 } }
fn row_index(i: usize, j: usize, digit: usize) -> usize { (i * 8 + j) * 15 + digit - 1 }
fn word(bytes: &[u8]) -> u64 { u64::from_le_bytes(bytes.try_into().expect("fixed word")) }

// Factory is private; production always uses actual OS entropy. Tests alone
// can replace this source, with no deterministic runtime feature or hook.
trait Entropy { fn fill(&mut self, bytes: &mut [u8]) -> Result<(), Error>; }
struct OsEntropy;
impl Entropy for OsEntropy {
    fn fill(&mut self, bytes: &mut [u8]) -> Result<(), Error> { getrandom::fill(bytes).map_err(|_| Error::Entropy) }
}

fn encrypt(id: &Arc<Anchor>, role: Role, secret: &[u8], message: u64, bytes: &[u8]) -> Ciphertext {
    let mut words = Zeroizing::new(vec![0u64; secret.len() + 1]);
    let guard = WipeGuard::new(&mut words);
    let mut b = message.wrapping_add(sign(bytes[secret.len() * 8]) as u64);
    for (i, &s) in secret.iter().enumerate() {
        let a = word(&bytes[i * 8..i * 8 + 8]);
        guard.words[i] = a;
        b = b.wrapping_add(a.wrapping_mul(s as u64));
    }
    guard.words[secret.len()] = b;
    guard.finish();
    Ciphertext { id: Arc::clone(id), role, words }
}

fn issue(entropy: &mut impl Entropy, max_bytes: usize, max_work: u64) -> Result<Session, Error> {
    let q = preflight(max_bytes, max_work)?; // before ALL material allocation and RNG
    let mut work = KernelWorkLimit::new(max_work);
    work.debit(q.work_units).map_err(|_| Error::Budget)?; // attempt charged, no refund/retry
    let id = Arc::new(Anchor { _live: 0 }); // nonzero allocation, unforgeable live pointer identity
    let mut random = Zeroizing::new(vec![0u8; REQUESTED_OS_BYTES]);
    if let Err(error) = entropy.fill(&mut random) {
        random.as_mut_slice().zeroize(); // observe real partial-fill failure path
        #[cfg(test)] tests::observe_wipe(5, random.len(), random.iter().all(|&x| x == 0));
        return Err(error);
    }
    let input = InputClient { id: Arc::clone(&id), secret: Zeroizing::new(random[..8].iter().map(|x| x & 1).collect()) };
    let output = OutputClient { id: Arc::clone(&id), secret: Zeroizing::new(random[8..12].iter().map(|x| x & 1).collect()), terminal: Cell::new(false) };
    let mut evaluator = Evaluator { id: Arc::clone(&id), table: Zeroizing::new(vec![0; ROWS * 5]), terminal: false };
    let mut oracle = Oracle { id: Arc::clone(&id), errors: Zeroizing::new(vec![0; ROWS]), input_errors: Zeroizing::new(vec![0; 4]), terminal: Cell::new(false) };
    let mut cursor = 12;
    for i in 0..8 { for j in 0..8 { for d in 1..16 {
        let r = row_index(i, j, d);
        let bytes = &random[cursor..cursor + 33]; cursor += 33;
        let message = (input.secret[i] as u64).wrapping_mul(d as u64) << (64 - BASE_LOG * (j + 1));
        let row = encrypt(&id, Role::Output, &output.secret, message, bytes);
        evaluator.table[r * 5..r * 5 + 5].copy_from_slice(&row.words);
        oracle.errors[r] = sign(bytes[32]);
        #[cfg(test)] tests::construction_checkpoint(r);
    } } }
    let mut ciphertexts = Vec::with_capacity(4);
    for m in 0..4 {
        let bytes = &random[cursor..cursor + 65]; cursor += 65;
        ciphertexts.push(encrypt(&id, Role::Input, &input.secret, (m as u64) << 62, bytes));
        oracle.input_errors[m] = sign(bytes[64]);
    }
    debug_assert_eq!(cursor, REQUESTED_OS_BYTES);
    Ok(Session { input, output, evaluator, oracle, ciphertexts, terminal: false })
}

impl Evaluator {
    fn switch(&mut self, ct: &Ciphertext) -> Result<Ciphertext, Error> {
        if self.terminal { return Err(Error::Terminal); }
        if !Arc::ptr_eq(&self.id, &ct.id) || ct.role != Role::Input || ct.words.len() != 9 {
            self.terminal = true;
            return Err(Error::Binding);
        }
        self.terminal = true; // unwind/error cannot reuse partially evaluated owner
        let mut out = Zeroizing::new(vec![0u64; 5]);
        let guard = WipeGuard::new(&mut out);
        guard.words[4] = ct.words[8];
        for i in 0..8 {
            let rounded = ct.words[i].wrapping_add(1u64 << 31);
            for j in 0..8 {
                let d = ((rounded >> (64 - BASE_LOG * (j + 1))) & 15) as usize;
                if d == 0 { continue; }
                let r = row_index(i, j, d);
                for k in 0..5 { guard.words[k] = guard.words[k].wrapping_sub(self.table[r * 5 + k]); }
            }
        }
        guard.finish();
        self.terminal = false;
        Ok(Ciphertext { id: Arc::clone(&self.id), role: Role::Output, words: out })
    }
}

impl OutputClient {
    fn phase(&self, ct: &Ciphertext) -> Result<u64, Error> {
        if self.terminal.get() { return Err(Error::Terminal); }
        if !Arc::ptr_eq(&self.id, &ct.id) || ct.role != Role::Output || ct.words.len() != 5 {
            self.terminal.set(true);
            return Err(Error::Binding);
        }
        Ok(ct.words[..4].iter().zip(self.secret.iter()).fold(ct.words[4], |b, (&a, &s)| b.wrapping_sub(a.wrapping_mul(s as u64))))
    }
}

// Independent oracle: UNBOUNDED quotient, signed residual, own quotient digit
// extraction and row indexing. Never calls production switch/decompose helpers.
fn quotient_residual(a: u64) -> (u128, i128) {
    let q = 1u128 << 32;
    let k = (a as u128 + q / 2) / q;
    (k, a as i128 - (k * q) as i128)
}
impl Oracle {
    fn expected(&self, input: &InputClient, ct: &Ciphertext, m: usize) -> Result<(u64, u64), Error> {
        if self.terminal.get() { return Err(Error::Terminal); }
        if m >= 4 || ct.role != Role::Input || ct.words.len() != 9 || !Arc::ptr_eq(&self.id, &input.id) || !Arc::ptr_eq(&self.id, &ct.id) {
            self.terminal.set(true);
            return Err(Error::Binding);
        }
        let mut error = self.input_errors[m] as i128;
        for i in 0..8 {
            let (k, residual) = quotient_residual(ct.words[i]);
            error += input.secret[i] as i128 * residual;
            let rounded = (k * (1u128 << 32)) % (1u128 << 64);
            for j in 0..8 {
                let power = 1u128 << (60 - 4 * j);
                let d = (rounded / power) % 16;
                if d != 0 { error -= self.errors[(i * 8 + j) * 15 + d as usize - 1] as i128; }
            }
        }
        Ok((((m as i128) * (1i128 << 62) + error).rem_euclid(1i128 << 64) as u64, error.unsigned_abs() as u64))
    }
}
fn decode(phase: u64) -> u64 { phase.wrapping_add(1u64 << 61) >> 62 }
fn centered_error(phase: u64, m: usize) -> u64 { (phase.wrapping_sub((m as u64) << 62) as i64).unsigned_abs() }

#[derive(Clone, Copy, Debug)]
pub struct Aggregate {
    pub decoded_matches: u64,
    pub phase_equalities: u64,
    pub maximum_centered_error: u64,
    pub issuance_ns: u64,
    pub switch_ns: u64,
    pub decrypt_ns: u64,
    pub oracle_ns: u64,
}
fn elapsed(start: Instant) -> u64 { start.elapsed().as_nanos().min(u64::MAX as u128) as u64 }

impl Session {
    fn finish(mut self, issuance_ns: u64) -> Result<Aggregate, Error> {
        if self.terminal { return Err(Error::Terminal); }
        self.terminal = true; // consuming facade: any failure drops every owner, no partial aggregate
        let mut aggregate = Aggregate { decoded_matches: 0, phase_equalities: 0, maximum_centered_error: 0,
            issuance_ns, switch_ns: 0, decrypt_ns: 0, oracle_ns: 0 };
        for (m, ct) in self.ciphertexts.iter().enumerate() {
            let start = Instant::now();
            let switched = self.evaluator.switch(ct)?;
            aggregate.switch_ns += elapsed(start);
            let start = Instant::now();
            let actual = self.output.phase(&switched)?;
            let decoded = decode(actual);
            aggregate.decrypt_ns += elapsed(start);
            let start = Instant::now();
            let (expected, magnitude) = self.oracle.expected(&self.input, ct, m)?;
            if actual != expected || magnitude != centered_error(actual, m) || magnitude > ERROR_BOUND { return Err(Error::Verification); }
            aggregate.phase_equalities += 1;
            if decoded != m as u64 { return Err(Error::Verification); }
            aggregate.decoded_matches += 1;
            aggregate.maximum_centered_error = aggregate.maximum_centered_error.max(magnitude);
            aggregate.oracle_ns += elapsed(start);
        }
        Ok(aggregate)
    }
}

/// One fresh session, reusable table, four public messages. All random fills and
/// input encryption charged to issuance wall time; call CPU measured by wrapper.
pub fn fixture() -> Result<Aggregate, Error> {
    preflight(MAX_LOGICAL_BYTES, u64::MAX)?;
    let start = Instant::now();
    let session = issue(&mut OsEntropy, MAX_LOGICAL_BYTES, u64::MAX)?;
    session.finish(elapsed(start))
}

#[cfg(test)]
mod tests;
