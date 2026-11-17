//! Candidate generic foundation for resolving immutable, single-successor paths.
//!
//! This is not protected-producer integration, PUMA coverage, activation, or a
//! speed benchmark. SourceNorms integration and constructor/publisher-V tariffs
//! remain held: index/memo/path birth, initialization, overlap, and retirement
//! still require funding under the original owner. No admission authority or
//! new accounting ledger is introduced here. The existing bounded index remains
//! separate and unchanged.
//!
//! The caller funds every storage element. Construction takes O(n) time and
//! queries collectively take O(n + q), including O(1) work per cached query.
//! Retained borrows prevent graph or scratch changes between queries.

/// One immutable producer relationship, independent of any model or method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Link {
    Terminal,
    Next(usize),
    Reject,
}

/// Failure category only; no query-dependent witness can become stale in cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResolveError {
    Missing,
    Rejected,
    Cycle,
}

/// Caller-provided memo storage; construction resets only the graph-sized prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MemoState {
    Unseen,
    Visiting,
    Done(Result<usize, ResolveError>),
}

/// At least one caller-provided scratch slice cannot hold the whole graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScratchTooSmall;

/// Owns exclusive scratch borrows for the lifetime of an immutable graph borrow.
///
/// Fields are private: callers cannot invalidate memoized results between queries.
pub(crate) struct Resolver<'a> {
    links: &'a [Link],
    memo: &'a mut [MemoState],
    path: &'a mut [usize],
    #[cfg(test)]
    visits: usize,
    #[cfg(test)]
    unwinds: usize,
}

/// Inactive logical storage and structural-work quote, not an admission token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolverScratchQuote {
    pub(crate) storage_bytes: usize,
    pub(crate) initialization_slots: usize,
    pub(crate) query_entries: usize,
    pub(crate) probe_steps: usize,
    pub(crate) link_steps: usize,
    pub(crate) unwind_steps: usize,
}

/// Quote a hypothetical caller owning three vectors, each with capacity EXACTLY
/// `n`: `Vec<Link>`, `Vec<MemoState>` and `Vec<usize>`. Each vector's `n` elements
/// is initialized once; `Resolver::new` then resets the memo in a fourth pass.
/// Thus initialization covers `4*n` slots, even if no queries are made.
///
/// Storage is the exact typed logical sum of the three vector headers, one
/// `Resolver<'static>` borrow holder and the three `n`-element backings. It is
/// NOT heap-only: inline headers and the borrow holder are included. The actual
/// compiled resolver layout is used, including test-only counters when present.
///
/// For at most `q` queries on one immutable `n`-node graph, structural upper
/// bounds are `q` query entries, `n+q` probe-loop iterations, `n` link steps and
/// `n` unwind steps. A probe means one `resolve` loop iteration, NOT every
/// comparison or load. Invalid-start work belongs to query entries, not probes.
///
/// Excess capacities, allocation transients, producer-map storage, link
/// construction, request owners and retirement are excluded. Resolver only
/// borrows these vectors; this quote implies neither three actual releases nor
/// any Drop behavior. Inputs are caller bounds, not trusted provenance or
/// admission authority. This neither funds the original owner nor wires
/// SourceNorms, publishers, runtime behavior or a new work ledger.
/// Returns `None` if any quoted arithmetic overflows.
pub(crate) fn resolver_scratch_quote(n: usize, q: usize) -> Option<ResolverScratchQuote> {
    let inline_bytes = std::mem::size_of::<Vec<Link>>()
        .checked_add(std::mem::size_of::<Vec<MemoState>>())?
        .checked_add(std::mem::size_of::<Vec<usize>>())?
        .checked_add(std::mem::size_of::<Resolver<'static>>())?;
    let slot_bytes = std::mem::size_of::<Link>()
        .checked_add(std::mem::size_of::<MemoState>())?
        .checked_add(std::mem::size_of::<usize>())?;
    Some(ResolverScratchQuote {
        storage_bytes: inline_bytes.checked_add(n.checked_mul(slot_bytes)?)?,
        initialization_slots: n.checked_mul(4)?,
        query_entries: q,
        probe_steps: n.checked_add(q)?,
        link_steps: n,
        unwind_steps: n,
    })
}

impl<'a> Resolver<'a> {
    /// Check both capacities before any mutation, then reset just the live memo.
    /// Path contents and both spare tails remain untouched by construction.
    pub(crate) fn new(
        links: &'a [Link],
        memo: &'a mut [MemoState],
        path: &'a mut [usize],
    ) -> Result<Self, ScratchTooSmall> {
        let n = links.len();
        if memo.len() < n || path.len() < n {
            return Err(ScratchTooSmall);
        }
        memo[..n].fill(MemoState::Unseen);
        Ok(Self {
            links,
            memo: &mut memo[..n],
            path: &mut path[..n],
            #[cfg(test)]
            visits: 0,
            #[cfg(test)]
            unwinds: 0,
        })
    }

    /// Resolve to the terminal node's index, or cache a failure category.
    ///
    /// Invalid starts take O(1) and do not touch scratch. An invalid edge caches
    /// `Missing` on its valid prefix. Every newly visited node is unwound to
    /// `Done` before returning, including cycle members and their incoming prefix.
    pub(crate) fn resolve(&mut self, start: usize) -> Result<usize, ResolveError> {
        if start >= self.links.len() {
            return Err(ResolveError::Missing);
        }

        let mut node = start;
        let mut depth = 0;
        let result = loop {
            // Every in-range node has a memo slot; an out-of-range edge exits
            // before indexing either slice.
            if node >= self.links.len() {
                break Err(ResolveError::Missing);
            }
            match self.memo[node] {
                MemoState::Done(result) => break result,
                MemoState::Visiting => break Err(ResolveError::Cycle),
                MemoState::Unseen => {}
            }

            // Only distinct Unseen nodes are pushed. Thus depth < n here,
            // and constructor preflight guarantees the path slot exists.
            self.memo[node] = MemoState::Visiting;
            self.path[depth] = node;
            depth += 1;
            #[cfg(test)]
            {
                self.visits += 1;
            }
            match self.links[node] {
                Link::Terminal => break Ok(node),
                Link::Next(next) => node = next,
                Link::Reject => break Err(ResolveError::Rejected),
            }
        };

        while depth != 0 {
            depth -= 1;
            self.memo[self.path[depth]] = MemoState::Done(result);
            #[cfg(test)]
            {
                self.unwinds += 1;
            }
        }
        result
    }
}

#[cfg(test)]
#[path = "semantic_producer_paths_tests.rs"]
mod tests;
