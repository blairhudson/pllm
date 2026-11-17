//! Owned string/plain-usize sequence logical retirement v1 (inactive).
//!
//! These numeric observations describe exposed logical endpoints, not private
//! Rust Drop instructions, allocator CPU/latency, RSS, or release acknowledgement.
//! No payload bytes or dimension values are read or wiped. The caller must
//! separately sponsor this query before borrowing the owners: `W` covers future
//! logical retirement ONLY, not evaluation, its errors, or the caller's tail.
//! Copies of a quote confer no custody, payment, source identity, or authority.
//! Ordinary Rust destruction is unchanged; no runtime obligation is installed.
//! Only allocation release is opaque: one public pointer/layout handoff, not a
//! successful resource-release acknowledgement. No arbitrary Drop callbacks are
//! supported. External unwind may enter retirement; panic runtime, unwinder
//! transport and cleanup completion under panic=abort are outside this profile.
//! Completed children/releases leave the logical cleanup suffix exactly once;
//! this describes accounting, not a runtime bitset or a private std drop order.
#![allow(clippy::ptr_arg)] // Deliberately closed concrete ownership profiles.

use super::{checked_array_bytes, Error};
use std::{alloc::Layout, mem::size_of};

#[path = "retirement_query_capacity.rs"]
mod query_capacity;
pub use query_capacity::*;

type Pair = (String, Vec<usize>);
type Sequence = Vec<Pair>;

/// Plain Unit1 call/visit/byte-potential counts and reusable nominal slot union.
/// `s` excludes owned backing and is not a physical stack/allocator bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LogicalRetirementWork {
    pub k: u64,
    pub v: u64,
    pub p: u64,
    pub s: u64,
}

impl LogicalRetirementWork {
    /// W = K + V + P. Does not pay for obtaining or inspecting this record.
    pub fn checked_total(self) -> Result<u64, Error> {
        add(add(self.k, self.v)?, self.p)
    }
}

/// Observed actual ownership, not a grant or a bound on future resident capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LogicalRetirementQuote {
    pub work: LogicalRetirementWork,
    /// Initialized outer pairs; zero for either standalone child profile.
    pub initialized_pairs: usize,
    pub allocation_releases: usize,
    /// Inline owner plus full exposed capacities (H_owner), not allocator size.
    pub h_owner: usize,
}

fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}

fn mul(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b).ok_or(Error::Overflow)
}

fn count(n: usize) -> Result<u64, Error> {
    u64::try_from(n).map_err(|_| Error::Overflow)
}

fn bytes_add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}

fn sum<const N: usize>(terms: [u64; N]) -> Result<u64, Error> {
    terms.into_iter().try_fold(0, add)
}

// Internal eligibility only: no generic Vec<T> or ZST-owning public profile.
fn allocation_facts(capacity: usize, element_size: usize) -> Result<(usize, usize), Error> {
    if capacity == 0 || element_size == 0 {
        return Ok((0, 0));
    }
    Ok((capacity.checked_mul(element_size).ok_or(Error::Overflow)?, 1))
}

impl LogicalRetirementWork {
    // Call(A,n): K=n, V=0, P=2*A*n; reusable S=A when active.
    fn call(&mut self, pool: u64, instances: u64) -> Result<(), Error> {
        self.k = add(self.k, instances)?;
        self.p = add(self.p, mul(mul(2, pool)?, instances)?)?;
        if instances != 0 {
            self.s = add(self.s, pool)?;
        }
        Ok(())
    }

    // Ctl(U,e,n): V=e*n, P=2*U*(e+1)*n; reusable S=U when active.
    fn control(&mut self, pool: u64, events: u64, instances: u64) -> Result<(), Error> {
        self.v = add(self.v, mul(events, instances)?)?;
        self.p = add(self.p, mul(mul(mul(2, pool)?, add(events, 1)?)?, instances)?)?;
        if instances != 0 {
            self.s = add(self.s, pool)?;
        }
        Ok(())
    }
}

fn call_pool<Argument, Return>() -> Result<u64, Error> {
    mul(2, add(count(size_of::<Argument>())?, count(size_of::<Return>())?)?)
}

#[derive(Clone, Copy)]
struct ChildRow {
    call: u64,
    control: u64,
    header: usize,
}

impl ChildRow {
    fn string() -> Result<Self, Error> {
        Ok(Self {
            call: call_pool::<&mut String, ()>()?,
            control: child_control_pool()?,
            header: size_of::<String>(),
        })
    }

    fn usize_vec() -> Result<Self, Error> {
        Ok(Self {
            call: call_pool::<&mut Vec<usize>, ()>()?,
            control: child_control_pool()?,
            header: size_of::<Vec<usize>>(),
        })
    }

    fn append(self, work: &mut LogicalRetirementWork, n: u64) -> Result<(), Error> {
        work.call(self.call, n)?;
        work.control(self.control, 6, n)?;
        work.p = add(work.p, mul(count(self.header)?, n)?)?;
        Ok(())
    }
}

fn child_control_pool() -> Result<u64, Error> {
    sum([
        mul(3, count(size_of::<usize>())?)?,
        count(size_of::<*mut u8>())?,
        count(size_of::<Layout>())?,
        count(size_of::<bool>())?,
    ])
}

fn release_rows(work: &mut LogicalRetirementWork, releases: u64) -> Result<(), Error> {
    work.call(call_pool::<(*mut u8, Layout), ()>()?, releases)?;
    work.control(sum([
        count(size_of::<usize>())?,
        count(size_of::<bool>())?,
        count(size_of::<()>())?,
    ])?, 3, releases)
}

fn child_quote(row: ChildRow, backing: usize, releases: usize) -> Result<LogicalRetirementQuote, Error> {
    let mut work = LogicalRetirementWork::default();
    row.append(&mut work, 1)?;
    release_rows(&mut work, count(releases)?)?;
    work.checked_total()?;
    Ok(LogicalRetirementQuote {
        work,
        initialized_pairs: 0,
        allocation_releases: releases,
        h_owner: bytes_add(row.header, backing)?,
    })
}

/// One concrete String shell and zero/one logical allocation-release endpoint.
pub fn string_retirement_v1(value: &String) -> Result<LogicalRetirementQuote, Error> {
    let (backing, releases) = allocation_facts(value.capacity(), size_of::<u8>())?;
    child_quote(ChildRow::string()?, backing, releases)
}

/// One concrete plain-usize Vec shell; no scalar-element retirement loop.
pub fn usize_vec_retirement_v1(value: &Vec<usize>) -> Result<LogicalRetirementQuote, Error> {
    let backing = checked_array_bytes::<usize>(value.capacity())?;
    let (_, releases) = allocation_facts(value.capacity(), size_of::<usize>())?;
    child_quote(ChildRow::usize_vec()?, backing, releases)
}

fn sequence_work(initialized: usize, releases: usize) -> Result<LogicalRetirementWork, Error> {
    let m = count(initialized)?;
    let successors = count(initialized.checked_add(1).ok_or(Error::Overflow)?)?;
    let u = count(size_of::<usize>())?;
    let b = count(size_of::<bool>())?;
    let mut work = LogicalRetirementWork::default();
    work.call(call_pool::<&mut Sequence, ()>()?, 1)?;
    work.control(sum([count(size_of::<&mut [Pair]>())?, mul(3, u)?, b])?, 8, 1)?;
    work.call(call_pool::<(&mut [Pair], usize), Option<&mut Pair>>()?, successors)?;
    work.control(add(u, b)?, 2, successors)?;
    work.call(call_pool::<&mut Pair, ()>()?, m)?;
    work.control(sum([u, count(size_of::<u8>())?, mul(2, b)?])?, 5, m)?;
    ChildRow::string()?.append(&mut work, m)?;
    ChildRow::usize_vec()?.append(&mut work, m)?;
    release_rows(&mut work, count(releases)?)?;
    // Child rows supplied their own header reads; this closes J with Seq's.
    work.p = add(work.p, count(size_of::<Sequence>())?)?;
    work.checked_total()?;
    Ok(work)
}

/// One borrowed pass over initialized owners, never payload or spare pair slots.
/// H_owner retains actual outer/child capacities; S is reusable control only.
/// Normal and externally unwinding entry share this single logical envelope.
/// With m initialized pairs and a allocated owners, K=2+4m+a, V=10+19m+3a.
/// P includes the complete typed event rows and owner-header reads; W=K+V+P.
pub fn string_usize_sequence_retirement_v1(value: &Vec<(String, Vec<usize>)>) -> Result<LogicalRetirementQuote, Error> {
    let mut backing = checked_array_bytes::<Pair>(value.capacity())?;
    let (_, mut releases) = allocation_facts(value.capacity(), size_of::<Pair>())?;
    for (name, dimensions) in value {
        let (name_bytes, name_release) = allocation_facts(name.capacity(), size_of::<u8>())?;
        let dimension_bytes = checked_array_bytes::<usize>(dimensions.capacity())?;
        let (_, dimension_release) = allocation_facts(dimensions.capacity(), size_of::<usize>())?;
        backing = bytes_add(backing, bytes_add(name_bytes, dimension_bytes)?)?;
        releases = bytes_add(releases, bytes_add(name_release, dimension_release)?)?;
    }
    Ok(LogicalRetirementQuote {
        work: sequence_work(value.len(), releases)?,
        initialized_pairs: value.len(),
        allocation_releases: releases,
        h_owner: bytes_add(size_of::<Sequence>(), backing)?,
    })
}

/// Future WORK only: a=2*m+1, including possibly allocated empty outer Vec.
/// Dominates initialized prefixes <=m; does not bound capacity, growth internals,
/// replacement overlap, or a not-yet-inserted local child. No prepaid grant.
pub fn string_usize_sequence_retirement_bound_v1(max_initialized_pairs: usize) -> Result<LogicalRetirementWork, Error> {
    let releases = max_initialized_pairs.checked_mul(2)
        .and_then(|n| n.checked_add(1)).ok_or(Error::Overflow)?;
    sequence_work(max_initialized_pairs, releases)
}

#[cfg(test)]
#[path = "retirement_tests.rs"]
mod tests;
