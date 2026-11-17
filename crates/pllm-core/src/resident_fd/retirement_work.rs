//! Conditional logical source-work envelope for one resident-FD retirement attempt.
//!
//! # Declaration, not qualification
//! The named profile is a **caller-declared conjunction of premises**, not an
//! authenticated proof or capability: this audited repository implementation and
//! Rust/std/alloc 1.85.1; macOS64 Apple mutex backend/TLS, non-Miri; concrete
//! `UnixCloseOnceFd` and `physical_close`; destination mutex **already warm at
//! entry**; root `cfg(test) = false`; `mir_emit_retag = false`;
//! `precise_enum_drop_elaboration = true`; ordinary non-coroutines; default
//! Global -> System allocator with no override; selected-target public leaf
//! sizes and private paired size/alignment bounds. Allocator qualification is
//! conditional, not a whole-workspace proof. Foreign-lease provenance does not
//! establish destination warmth. Never warm or lock a registry to inspect it.
//!
//! Visible unsupported target, root test, Miri, or unqualified input is rejected
//! before arithmetic. Compiler version, library flags, retag setting, allocator,
//! and warmth cannot be introspected here. Non-Miri does not imply no retag.
//! Std poison alpha, std debug delta-S, alloc debug delta-A, and TSan tau are
//! independent selected-library flags, never inferred from root debug or panic
//! settings. All 16 combinations and compatible paths are maximized, covering
//! both root panic strategies independently of alpha. Componentwise maxima may
//! come from different paths: this is a conservative envelope, not one trace.
//!
//! Units are normalized/elaborated source helper-call occurrences, visits, and
//! potential bytes, not machine events. Excludes physical CPU/RSS, native TLS
//! bootstrap, pthread blocking, foreign allocator, close/free/abort internals,
//! runtime Display/panic/hooks and unwinder transport. Only source call
//! occurrences at those boundaries are counted. There is no retained-FD count,
//! R(N), storage charge, close acknowledgement, security, funding, admission, or
//! activation claim. Borrowed excludes a future external caller Option drop;
//! consuming includes its local caller drop. Each retry is a new attempt.
//! Computing this quote is not funded by the quoted attempt; future bootstrap
//! cost must be separate. No registry/reference, lock, allocation, acquisition,
//! preparation, or physical I/O is used by this computation.
//!
//! # Source contract references (rust-lang/rust, tag 1.85.1)
//! `alloc/src/sync.rs` (Arc clone/drop);
//! `std/sys/sync/mutex/pthread.rs` (fail/lock/unlock);
//! `compiler/rustc_mir_build/src/builder/scope.rs:1633..1645`
//! (build_unwind_tree); `rustc_mir_transform/src/shim.rs` (static drop, no flags),
//! `remove_noop_landing_pads.rs` (cleanup Continue), `elaborate_drops.rs` and
//! `rustc_mir_dataflow/src/impls/initialized.rs` (moved/initialized states and
//! inactive variants); `rustc_session/src/options.rs` (precise enum drop default
//! true, MIR retag default false). These support a conditional source contract,
//! not a native execution or panic-cost claim.

use super::{FileLease, RegistryError};
use crate::fd_close_once::{CloseFailure, CloseState, UnixCloseOnceFd};
use core::{
    alloc::Layout, cell::Cell, convert::Infallible, fmt, mem::size_of, ops::ControlFlow,
    sync::atomic::Ordering,
};
use std::{
    io,
    thread::{AccessError, LocalKey},
};

/// Caller declaration only; selecting a variant verifies none of its premises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceProfile {
    Unqualified,
    Rust1851Mac64WarmSystemNoRetag,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetirementEntry {
    Borrowed,
    Consuming,
}

/// Descriptive logical envelope, with no payment or capability authority.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConditionalRetirementWorkQuote {
    pub helper_calls: u64,
    pub visits: u64,
    pub potential_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuoteError {
    UnsupportedProfile,
    Overflow,
}

impl ConditionalRetirementWorkQuote {
    pub fn total_work(self) -> Result<u64, QuoteError> {
        linear(
            0,
            &[
                (1, self.helper_calls),
                (1, self.visits),
                (1, self.potential_bytes),
            ],
        )
    }

    fn add(self, other: Self) -> Result<Self, QuoteError> {
        Ok(work(
            linear(self.helper_calls, &[(1, other.helper_calls)])?,
            linear(self.visits, &[(1, other.visits)])?,
            linear(self.potential_bytes, &[(1, other.potential_bytes)])?,
        ))
    }

    fn include(&mut self, candidate: Self) -> Result<(), QuoteError> {
        // A compatible branch must not disappear merely because another branch
        // dominates one component. Validate even its final h + v + b.
        candidate.total_work()?;
        self.helper_calls = self.helper_calls.max(candidate.helper_calls);
        self.visits = self.visits.max(candidate.visits);
        self.potential_bytes = self.potential_bytes.max(candidate.potential_bytes);
        Ok(())
    }
}

type Work = ConditionalRetirementWorkQuote;

const fn work(helper_calls: u64, visits: u64, potential_bytes: u64) -> Work {
    Work {
        helper_calls,
        visits,
        potential_bytes,
    }
}

fn linear(base: u64, terms: &[(u64, u64)]) -> Result<u64, QuoteError> {
    terms.iter().try_fold(base, |sum, &(coefficient, value)| {
        sum.checked_add(coefficient.checked_mul(value).ok_or(QuoteError::Overflow)?)
            .ok_or(QuoteError::Overflow)
    })
}

fn size_u64(value: usize) -> Result<u64, QuoteError> {
    u64::try_from(value).map_err(|_| QuoteError::Overflow)
}

fn leaf<T>() -> Result<u64, QuoteError> {
    size_u64(size_of::<T>())
}

/// No caller-supplied layout, price, path counts, or registry is accepted.
pub fn quote(
    profile: SourceProfile,
    entry: RetirementEntry,
) -> Result<ConditionalRetirementWorkQuote, QuoteError> {
    if profile != SourceProfile::Rust1851Mac64WarmSystemNoRetag
        || !cfg!(all(
            target_os = "macos",
            target_pointer_width = "64",
            not(test),
            not(miri)
        ))
    {
        return Err(QuoteError::UnsupportedProfile);
    }
    envelope(entry, Leaves::actual()?)
}

// Actual PUBLIC leaves only. No private standard-library layout is inferred.
#[derive(Clone, Copy)]
struct Leaves {
    e: u64,
    rr: u64,
    d: u64,
    f: u64,
    of: u64,
    q: u64,
    a9: u64,
    f9: u64,
    h: u64,
    r_public: u64,
    l: u64,
    p: u64,
    o: u64,
    bs: u64,
    ds: u64,
    afmt: u64,
    aopt: u64,
    eio: u64,
}

impl Leaves {
    fn actual() -> Result<Self, QuoteError> {
        type Tc = Cell<(usize, bool)>;
        Ok(Self {
            e: leaf::<RegistryError>()?,
            rr: leaf::<Result<Infallible, RegistryError>>()?,
            d: leaf::<Option<UnixCloseOnceFd>>()?,
            f: leaf::<FileLease>()?,
            of: leaf::<Option<FileLease>>()?,
            q: leaf::<Result<(), RegistryError>>()?,
            a9: leaf::<CloseFailure>()?,
            f9: linear(
                0,
                &[
                    (9, leaf::<Result<(), CloseFailure>>()?),
                    (4, leaf::<Result<(), ()>>()?),
                    (10, leaf::<CloseState>()?),
                    (6, leaf::<Result<(), i32>>()?),
                ],
            )?,
            h: linear(
                0,
                &[
                    (6, leaf::<(*const u8,)>()?),
                    (6, leaf::<(*const (), usize, bool)>()?),
                ],
            )?,
            r_public: leaf::<Result<usize, usize>>()?,
            l: leaf::<Layout>()?,
            p: leaf::<(usize, usize)>()?,
            o: leaf::<Ordering>()?,
            bs: linear(
                1128,
                &[
                    (2, leaf::<LocalKey<Tc>>()?),
                    (3, leaf::<Option<&'static Tc>>()?),
                    (7, leaf::<Result<&'static Tc, AccessError>>()?),
                    (
                        5,
                        leaf::<ControlFlow<Result<Infallible, AccessError>, &'static Tc>>()?,
                    ),
                    (7, leaf::<Result<bool, AccessError>>()?),
                    (5, leaf::<(usize, bool)>()?),
                    (8, leaf::<u128>()?),
                ],
            )?,
            ds: linear(108, &[(4, leaf::<(&'static i32, &'static i32)>()?)])?,
            afmt: leaf::<fmt::Arguments<'static>>()?,
            aopt: leaf::<Option<fmt::Arguments<'static>>>()?,
            eio: leaf::<io::Error>()?,
        })
    }
}

#[derive(Clone, Copy)]
struct Flags {
    alpha: u64,
    ds: u64,
    da: u64,
    tau: u64,
}

impl Flags {
    fn from_bits(bits: u8) -> Self {
        Self {
            alpha: u64::from(bits & 1),
            ds: u64::from((bits >> 1) & 1),
            da: u64::from((bits >> 2) & 1),
            tau: u64::from((bits >> 3) & 1),
        }
    }
}

struct Derived {
    b: u64,
    m: u64,
    c: u64,
    r: u64,
    k: u64,
    j_sum: u64,
    jp: u64,
    g: u64,
    pm: u64,
}

impl Derived {
    fn new(s: Leaves, flags: Flags) -> Result<Self, QuoteError> {
        let b = linear(14, &[(1, s.d)])?;
        let m = b.max(8);
        let c = linear(44, &[(1, m)])?;
        // These r/j/k/t are paired private layout bounds, NOT path counts.
        // Preserve these formulas; do not tighten to inferred private sizes.
        let r = linear(22, &[(1, s.e.max(8))])?;
        let j = linear(22, &[(1, s.e.max(30))])?;
        let k = linear(22, &[(1, s.rr.max(8))])?;
        let t = linear(22, &[(1, s.rr.max(30))])?;
        Ok(Self {
            b,
            m,
            c,
            r,
            k,
            j_sum: linear(0, &[(18, r), (10, k), (9, j), (5, t)])?,
            jp: linear(0, &[(9, r), (5, k), (9, j), (5, t), (9, s.rr)])?,
            g: linear(
                0,
                &[
                    (1293, flags.alpha),
                    (linear(4, &[(6, flags.alpha)])?, s.o),
                    (linear(0, &[(2, flags.alpha)])?, s.bs),
                    (flags.ds, s.ds),
                ],
            )?,
            pm: linear(0, &[(10, s.o), (2, s.bs), (flags.ds, s.ds)])?,
        })
    }
}

const DN: Work = work(10, 42, 940);
const E0: Work = work(10, 51, 1006);
const RFAIL: Work = work(0, 1, 0);
const RRETIRE: Work = work(0, 1, 0);
const RRG: Work = work(0, 1, 0);

fn last_arc(s: Leaves, d: &Derived, flags: Flags) -> Result<Work, QuoteError> {
    Ok(work(
        linear(97, &[(9, flags.tau), (6, flags.da)])?,
        linear(602, &[(49, flags.tau), (46, flags.da)])?,
        linear(
            7926,
            &[
                (1224, flags.tau),
                (870, flags.da),
                (1, d.b),
                (28, s.l),
                (2, s.p),
            ],
        )?,
    ))
}

// Each normal row pairs with exactly one compatible unlock envelope. No second
// custom RG drop, mutex death/unlock, or TLS store is appended. poison.done
// precedes the assertion; born-unwinding panic short-circuits. Second panic may
// truncate cleanup: this envelope does not assert continuation was executed.
fn normal_path(
    entry: RetirementEntry,
    flags: Flags,
    normal: Work,
    finish: bool,
    s: Leaves,
    emit: &mut impl FnMut(Work) -> Result<(), QuoteError>,
) -> Result<(), QuoteError> {
    emit(normal)?;
    if flags.ds == 1 {
        let assertion = work(0, 10, linear(128, &[(4, s.aopt)])?);
        let mut failure = normal.add(assertion)?;
        if finish {
            failure = failure.add(RRG)?;
        }
        if entry == RetirementEntry::Consuming {
            failure = failure.add(RRETIRE)?;
        }
        emit(failure)?;
    }
    Ok(())
}

fn paths(
    entry: RetirementEntry,
    s: Leaves,
    flags: Flags,
    mut emit: impl FnMut(Work) -> Result<(), QuoteError>,
) -> Result<(), QuoteError> {
    let d = Derived::new(s, flags)?;
    let consuming = entry == RetirementEntry::Consuming;
    let dl = last_arc(s, &d, flags)?;
    if !consuming {
        // Borrowed None has no mutex guard and therefore no unlock assertion.
        emit(work(
            9,
            13,
            linear(
                636,
                &[
                    (1, s.of),
                    (4, s.q),
                    (18, s.e),
                    (9, d.r),
                    (5, d.k),
                    (9, s.rr),
                ],
            )?,
        ))?;
    }
    // Common entry-specific terms for the early normal rows.
    let early = if consuming {
        linear(0, &[(2, s.f), (6, s.q)])?
    } else {
        linear(0, &[(1, s.of), (4, s.q)])?
    };
    let visit_adjust = if consuming { 0 } else { 1 };
    // Poison rows exist ONLY for alpha=1. No RG exists here. map_err's
    // Result<MG, RegistryError> is not generally no-Drop: on the panic path its
    // result is uninitialized, Err PE argument moved through ClosureOnceShim,
    // noncapturing environment no-Drop, active PE already being destroyed, and
    // inactive Ok MG not live. Precise enum drop => no extra map_err/shim unwind.
    if flags.alpha == 1 {
        let row = work(
            123,
            linear(359, &[(1, visit_adjust), (15, flags.ds)])?,
            linear(
                if consuming { 8270 } else { 8106 },
                &[(1, early), (24, s.e), (1, d.jp), (1, d.pm)],
            )?,
        );
        early_path(entry, flags, row, s, dl, &mut emit)?;
    }
    let j_common = linear(0, &[(1, early), (1, d.j_sum), (1, s.h), (1, d.g)])?;
    let bounds = work(
        linear(114, &[(56, flags.alpha)])?,
        linear(
            584,
            &[(1, visit_adjust), (202, flags.alpha), (15, flags.ds)],
        )?,
        linear(
            if consuming { 10527 } else { 10363 },
            &[(1, j_common), (22, s.e), (9, s.rr)],
        )?,
    );
    early_path(entry, flags, bounds, s, dl, &mut emit)?;
    for (helpers, visits, borrowed_bytes, consuming_bytes) in [
        (115, 615, 10665, 10829), // wrong custody
        (121, 642, 11251, 11415), // wrong Arc
        (125, 648, 11515, 11679), // phase
    ] {
        let row = work(
            linear(helpers, &[(56, flags.alpha)])?,
            linear(
                visits,
                &[(1, visit_adjust), (202, flags.alpha), (15, flags.ds)],
            )?,
            linear(
                if consuming {
                    consuming_bytes
                } else {
                    borrowed_bytes
                },
                &[(1, j_common), (2, d.c), (10, s.e)],
            )?,
        );
        early_path(entry, flags, row, s, dl, &mut emit)?;
    }
    let busy_terms = if consuming {
        linear(0, &[(2, s.f), (6, s.q)])?
    } else {
        linear(0, &[(12, s.f), (39, s.of), (4, s.q)])?
    };
    let busy = work(
        linear(if consuming { 646 } else { 640 }, &[(56, flags.alpha)])?,
        linear(
            if consuming { 4764 } else { 4728 },
            &[(202, flags.alpha), (15, flags.ds)],
        )?,
        linear(
            if consuming { 53075 } else { 50734 },
            &[
                (140, d.c),
                (1, busy_terms),
                (10, s.e),
                (1, d.j_sum),
                (15, s.h),
                (1, s.r_public),
                (4, d.m),
                (1, d.g),
            ],
        )?,
    );
    // Busy consuming already includes final DN; never append DN or DL again.
    normal_path(entry, flags, busy, true, s, &mut emit)?;
    let u = unique_terms(entry, s, &d, flags)?;
    let v = linear(
        if consuming { 5667 } else { 5671 },
        &[
            (202, flags.alpha),
            (49, flags.tau),
            (23, flags.da),
            (15, flags.ds),
        ],
    )?;
    for error in [false, true] {
        let h = match (consuming, error) {
            (false, false) => 773,
            (false, true) => 776,
            (true, false) => 769,
            (true, true) => 772,
        };
        // Corrected consuming constants: 61190/61270, NOT 60156/60236.
        let base = match (consuming, error) {
            (false, false) => 60028,
            (false, true) => 60108,
            (true, false) => 61190,
            (true, true) => 61270,
        };
        let row = work(
            linear(h, &[(56, flags.alpha), (9, flags.tau), (3, flags.da)])?,
            v,
            linear(
                base,
                &[
                    (1, u),
                    (if error { 10 } else { 8 }, s.e),
                    (if error { 12 } else { 0 }, s.a9),
                ],
            )?,
        );
        // Unique already contains WG cleanup. No WR/DN/DL for this same Arc.
        normal_path(entry, flags, row, true, s, &mut emit)?;
    }
    // PAL lock failure is BEFORE MG/RG. Efull = E0 + Rfail: never add Rfail
    // twice. W8 owns normal Error drop then Resume. W6 borrowed MG/poison bool
    // has no Drop; single-field PE adds zero Continue. W1 None wrapper's normal
    // return is already counted, not an additional Resume.
    let prefix = work(
        linear(if consuming { 61 } else { 63 }, &[(7, flags.ds)])?,
        linear(if consuming { 318 } else { 321 }, &[(58, flags.ds)])?,
        linear(
            if consuming { 5949 } else { 5856 },
            &[
                (if consuming { 2 } else { 0 }, s.f),
                (if consuming { 0 } else { 1 }, s.of),
                (4, s.e),
                (9, d.r),
                (5, d.k),
                (2, s.o),
                (8, s.eio),
                (8, s.afmt),
                (1014, flags.ds),
            ],
        )?,
    );
    emit(prefix)?; // aborting root: no unwind cleanup
    let cleanup = prefix.add(E0.add(RFAIL)?)?;
    if consuming {
        let cleanup = cleanup.add(work(2, 2, 155))?.add(RRETIRE)?;
        emit(cleanup.add(DN)?)?;
        emit(cleanup.add(dl)?)?;
    } else {
        emit(cleanup)?;
    }
    // Clone-abort: CLabort=(11,42,972), including intrinsic abort()->!
    // (1,0,0), no args/result/Never-size/cleanup. CLnormal=(12,48,1108)
    // dominates by (1,6,136). Whole W3 Busy normal=(40,174,3966+R+4M)
    // dominates abort=(39,168,3830+R+4M). Remaining normal allowances are
    // positive, so the already-emitted Busy row covers the abort prefix.
    Ok(())
}

fn early_path(
    entry: RetirementEntry,
    flags: Flags,
    base: Work,
    s: Leaves,
    dl: Work,
    emit: &mut impl FnMut(Work) -> Result<(), QuoteError>,
) -> Result<(), QuoteError> {
    if entry == RetirementEntry::Consuming {
        // One Arc-final alternative, never their sum. Consuming None unreachable.
        for drop in [DN, dl] {
            normal_path(entry, flags, base.add(drop)?, false, s, emit)?;
        }
    } else {
        normal_path(entry, flags, base, false, s, emit)?;
    }
    Ok(())
}

fn unique_terms(
    entry: RetirementEntry,
    s: Leaves,
    d: &Derived,
    flags: Flags,
) -> Result<u64, QuoteError> {
    let caller = if entry == RetirementEntry::Consuming {
        linear(0, &[(2, s.f), (6, s.q)])?
    } else {
        linear(0, &[(10, s.f), (34, s.of), (4, s.q)])?
    };
    linear(
        0,
        &[
            (142, d.c),
            (1, caller),
            (1, d.j_sum),
            (15, s.h),
            (1, s.d),
            (16, d.b),
            (1, s.r_public),
            (4, d.m),
            (24, s.l),
            (2, s.p),
            (1224, flags.tau),
            (435, flags.da),
            (1, d.g),
            (1, s.f9),
        ],
    )
}

fn envelope(entry: RetirementEntry, leaves: Leaves) -> Result<Work, QuoteError> {
    let mut maximum = Work::default();
    for bits in 0..16 {
        paths(entry, leaves, Flags::from_bits(bits), |candidate| {
            maximum.include(candidate)
        })?;
    }
    maximum.total_work()?;
    Ok(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dominates(a: Work, b: Work) -> bool {
        a.helper_calls >= b.helper_calls
            && a.visits >= b.visits
            && a.potential_bytes >= b.potential_bytes
    }

    #[test]
    fn unit_root_never_claims_profile_qualification() {
        for entry in [RetirementEntry::Borrowed, RetirementEntry::Consuming] {
            assert_eq!(
                quote(SourceProfile::Rust1851Mac64WarmSystemNoRetag, entry),
                Err(QuoteError::UnsupportedProfile)
            );
            assert_eq!(
                quote(SourceProfile::Unqualified, entry),
                Err(QuoteError::UnsupportedProfile)
            );
        }
    }

    #[test]
    fn checked_products_sums_and_final_total() {
        assert_eq!(linear(0, &[(2, u64::MAX)]), Err(QuoteError::Overflow));
        assert_eq!(linear(1, &[(1, u64::MAX)]), Err(QuoteError::Overflow));
        assert_eq!(work(u64::MAX, 1, 0).total_work(), Err(QuoteError::Overflow));
        assert_eq!(
            work(0, u64::MAX, 1).add(work(0, 1, 0)),
            Err(QuoteError::Overflow)
        );
        assert_eq!(
            size_u64(usize::MAX),
            u64::try_from(usize::MAX).map_err(|_| QuoteError::Overflow)
        );
    }

    #[test]
    fn every_compatible_branch_is_checked_even_if_dominated() {
        let mut maximum = work(u64::MAX, u64::MAX, u64::MAX);
        assert_eq!(
            maximum.include(work(u64::MAX, 1, 0)),
            Err(QuoteError::Overflow)
        );
        let mut leaves = Leaves::actual().unwrap();
        leaves.aopt = u64::MAX; // only an enabled unlock-assert branch uses this
        assert_eq!(
            envelope(RetirementEntry::Borrowed, leaves),
            Err(QuoteError::Overflow)
        );
        leaves = Leaves::actual().unwrap();
        leaves.l = u64::MAX; // consuming early DL cannot be silently skipped
        assert_eq!(
            envelope(RetirementEntry::Consuming, leaves),
            Err(QuoteError::Overflow)
        );
    }

    #[test]
    fn aggregate_dominates_every_path_in_all_sixteen_configurations() {
        let s = Leaves::actual().unwrap();
        for entry in [RetirementEntry::Borrowed, RetirementEntry::Consuming] {
            let max = envelope(entry, s).unwrap();
            for bits in 0..16 {
                let mut count = 0;
                paths(entry, s, Flags::from_bits(bits), |row| {
                    assert!(dominates(max, row));
                    count += 1;
                    Ok(())
                })
                .unwrap();
                assert!(count > 0);
            }
        }
    }

    #[test]
    fn positive_leaf_bounds_never_reduce_envelope() {
        let s = Leaves::actual().unwrap();
        // Independent increases, not an assumption about semantic flag toggles.
        for index in 0..18 {
            let mut larger = s;
            let fields = [
                &mut larger.e,
                &mut larger.rr,
                &mut larger.d,
                &mut larger.f,
                &mut larger.of,
                &mut larger.q,
                &mut larger.a9,
                &mut larger.f9,
                &mut larger.h,
                &mut larger.r_public,
                &mut larger.l,
                &mut larger.p,
                &mut larger.o,
                &mut larger.bs,
                &mut larger.ds,
                &mut larger.afmt,
                &mut larger.aopt,
                &mut larger.eio,
            ];
            *fields.into_iter().nth(index).unwrap() += 1;
            for entry in [RetirementEntry::Borrowed, RetirementEntry::Consuming] {
                assert!(dominates(
                    envelope(entry, larger).unwrap(),
                    envelope(entry, s).unwrap()
                ));
            }
        }
    }

    #[test]
    fn consuming_unique_corrected_constants_independent_hand_add() {
        let s = Leaves::actual().unwrap();
        let flags = Flags::from_bits(0);
        let d = Derived::new(s, flags).unwrap();
        // Independent literal hand addition of UC; intentionally not unique_terms.
        let uc = 142 * d.c
            + 2 * s.f
            + 6 * s.q
            + d.j_sum
            + 15 * s.h
            + s.d
            + 16 * d.b
            + s.r_public
            + 4 * d.m
            + 24 * s.l
            + 2 * s.p
            + d.g
            + s.f9;
        let mut ok = false;
        let mut err = false;
        paths(RetirementEntry::Consuming, s, flags, |row| {
            if row.helper_calls == 769 {
                assert_eq!(row.potential_bytes, 61190 + 8 * s.e + uc);
                ok = true;
            }
            if row.helper_calls == 772 {
                assert_eq!(row.potential_bytes, 61270 + 10 * s.e + uc + 12 * s.a9);
                err = true;
            }
            Ok(())
        })
        .unwrap();
        assert!(ok && err);
    }

    #[test]
    fn named_resumes_and_lock_cleanup_exactly_once() {
        assert_eq!(E0.add(RFAIL).unwrap(), work(10, 52, 1006));
        assert_eq!(RRETIRE, work(0, 1, 0));
        assert_eq!(RRG, work(0, 1, 0));
        assert_eq!(
            E0.add(RFAIL)
                .unwrap()
                .add(work(2, 2, 155))
                .unwrap()
                .add(RRETIRE)
                .unwrap(),
            work(12, 55, 1161)
        );
        let s = Leaves::actual().unwrap();
        let flags = Flags::from_bits(3);
        let base = work(100, 200, 300);
        for entry in [RetirementEntry::Borrowed, RetirementEntry::Consuming] {
            for finish in [false, true] {
                let mut seen = Vec::new();
                normal_path(entry, flags, base, finish, s, &mut |row| {
                    seen.push(row);
                    Ok(())
                })
                .unwrap();
                let resumes = u64::from(finish) + u64::from(entry == RetirementEntry::Consuming);
                assert_eq!(seen, [base, work(100, 210 + resumes, 428 + 4 * s.aopt)]);
            }
        }
    }

    #[test]
    fn one_arc_final_alternative_not_both_and_borrowed_external_drop_excluded() {
        let s = Leaves::actual().unwrap();
        let flags = Flags::from_bits(0);
        let d = Derived::new(s, flags).unwrap();
        let dl = last_arc(s, &d, flags).unwrap();
        let base = work(123, 359, 8270);
        let mut rows = Vec::new();
        early_path(RetirementEntry::Consuming, flags, base, s, dl, &mut |row| {
            rows.push(row);
            Ok(())
        })
        .unwrap();
        assert_eq!(rows, [base.add(DN).unwrap(), base.add(dl).unwrap()]);
        rows.clear();
        early_path(RetirementEntry::Borrowed, flags, base, s, dl, &mut |row| {
            rows.push(row);
            Ok(())
        })
        .unwrap();
        assert_eq!(rows, [base]); // future external Option drop not part of attempt
    }

    #[test]
    fn poison_only_when_enabled_and_consuming_none_unreachable() {
        let s = Leaves::actual().unwrap();
        for bits in 0..16 {
            let flags = Flags::from_bits(bits);
            let mut poison = 0;
            paths(RetirementEntry::Borrowed, s, flags, |row| {
                if row.helper_calls == 123 {
                    poison += 1;
                }
                Ok(())
            })
            .unwrap();
            assert_eq!(
                poison,
                if flags.alpha == 0 {
                    0
                } else if flags.ds == 0 {
                    1
                } else {
                    2
                }
            );
            paths(RetirementEntry::Consuming, s, flags, |row| {
                assert!(row.helper_calls > 9);
                Ok(())
            })
            .unwrap();
        }
    }

    #[test]
    fn unique_wg_and_busy_final_dn_have_no_extra_arc_drop() {
        let s = Leaves::actual().unwrap();
        let flags = Flags::from_bits(0);
        let mut helpers = Vec::new();
        paths(RetirementEntry::Consuming, s, flags, |row| {
            helpers.push(row.helper_calls);
            Ok(())
        })
        .unwrap();
        assert_eq!(helpers.iter().filter(|&&h| h == 646).count(), 1);
        assert_eq!(helpers.iter().filter(|&&h| h == 769).count(), 1);
        assert_eq!(helpers.iter().filter(|&&h| h == 772).count(), 1);
        assert!(!helpers.contains(&779)); // unique OK + DN would be wrong
        assert!(!helpers.contains(&866)); // unique OK + DL would be wrong
    }

    #[test]
    fn clone_abort_is_dominated_without_never_layout_or_cleanup() {
        let abort = work(11, 42, 972);
        let normal = work(12, 48, 1108);
        assert_eq!(abort.add(work(1, 6, 136)).unwrap(), normal);
        let s = Leaves::actual().unwrap();
        let d = Derived::new(s, Flags::from_bits(0)).unwrap();
        let extra = s.r_public + 4 * d.m;
        let w3_abort = work(39, 168, 3830 + extra);
        let w3_normal = work(40, 174, 3966 + extra);
        assert_eq!(w3_abort.add(work(1, 6, 136)).unwrap(), w3_normal);
        for entry in [RetirementEntry::Borrowed, RetirementEntry::Consuming] {
            assert!(dominates(envelope(entry, s).unwrap(), w3_abort));
        }
    }
}
