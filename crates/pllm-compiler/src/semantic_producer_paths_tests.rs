use super::{Link, MemoState, ResolveError, Resolver, ScratchTooSmall};

#[test]
fn scratch_quote_zero() {
    let quote = super::resolver_scratch_quote(0, 0).unwrap();
    assert_eq!(
        quote.storage_bytes,
        std::mem::size_of::<Vec<Link>>()
            + std::mem::size_of::<Vec<MemoState>>()
            + std::mem::size_of::<Vec<usize>>()
            + std::mem::size_of::<Resolver<'static>>()
    );
    assert_eq!(quote.initialization_slots, 0);
    assert_eq!(quote.query_entries, 0);
    assert_eq!(quote.probe_steps, 0);
    assert_eq!(quote.link_steps, 0);
    assert_eq!(quote.unwind_steps, 0);
}

#[test]
fn scratch_quote_exact() {
    for n in 0..=8 {
        for q in 0..=8 {
            let quote = super::resolver_scratch_quote(n, q).unwrap();
            // Independent typed formula, including the actual cfg(test) holder.
            let expected_storage = std::mem::size_of::<Vec<Link>>()
                + std::mem::size_of::<Vec<MemoState>>()
                + std::mem::size_of::<Vec<usize>>()
                + std::mem::size_of::<Resolver<'static>>()
                + n * (std::mem::size_of::<Link>()
                    + std::mem::size_of::<MemoState>()
                    + std::mem::size_of::<usize>());
            assert_eq!(quote.storage_bytes, expected_storage);
            assert_eq!(quote.initialization_slots, 4 * n);
            assert_eq!(quote.query_entries, q);
            assert_eq!(quote.probe_steps, n + q);
            assert_eq!(quote.link_steps, n);
            assert_eq!(quote.unwind_steps, n);
        }
    }
}

#[test]
fn scratch_quote_overflow() {
    let inline_bytes = std::mem::size_of::<Vec<Link>>()
        + std::mem::size_of::<Vec<MemoState>>()
        + std::mem::size_of::<Vec<usize>>()
        + std::mem::size_of::<Resolver<'static>>();
    let slot_bytes = std::mem::size_of::<Link>()
        + std::mem::size_of::<MemoState>()
        + std::mem::size_of::<usize>();
    // Pure arithmetic: no large vectors, arrays or allocator probes.
    assert_eq!(super::resolver_scratch_quote(usize::MAX, 0), None);
    let multiplication_overflow = (usize::MAX / slot_bytes).checked_add(1).unwrap();
    assert!(multiplication_overflow.checked_mul(slot_bytes).is_none());
    assert_eq!(super::resolver_scratch_quote(multiplication_overflow, 0), None);

    let storage_limit = usize::MAX.checked_sub(inline_bytes).unwrap() / slot_bytes;
    assert!(super::resolver_scratch_quote(storage_limit, 0).is_some());
    let addition_overflow = storage_limit.checked_add(1).unwrap();
    assert!(addition_overflow
        .checked_mul(slot_bytes)
        .and_then(|backing| inline_bytes.checked_add(backing))
        .is_none());
    assert_eq!(super::resolver_scratch_quote(addition_overflow, 0), None);

    // Initialization overflow is masked by the stricter typed storage bound;
    // its input still must be rejected rather than wrapping the four passes.
    let initialization_overflow = (usize::MAX / 4).checked_add(1).unwrap();
    assert!(initialization_overflow.checked_mul(4).is_none());
    assert_eq!(super::resolver_scratch_quote(initialization_overflow, 0), None);

    // Probe overflow is reachable while storage and initialization still fit.
    assert_eq!(super::resolver_scratch_quote(1, usize::MAX), None);
    let empty = super::resolver_scratch_quote(0, usize::MAX).unwrap();
    assert_eq!(empty.query_entries, usize::MAX);
    assert_eq!(empty.probe_steps, usize::MAX);
    let edge = super::resolver_scratch_quote(1, usize::MAX - 1).unwrap();
    assert_eq!(edge.probe_steps, usize::MAX);
}

#[test]
fn scratch_quote_monotonicity() {
    for n in 0..=8 {
        for q in 0..=8 {
            let base = super::resolver_scratch_quote(n, q).unwrap();
            for next in [
                super::resolver_scratch_quote(n + 1, q).unwrap(),
                super::resolver_scratch_quote(n, q + 1).unwrap(),
            ] {
                assert!(next.storage_bytes >= base.storage_bytes);
                assert!(next.initialization_slots >= base.initialization_slots);
                assert!(next.query_entries >= base.query_entries);
                assert!(next.probe_steps >= base.probe_steps);
                assert!(next.link_steps >= base.link_steps);
                assert!(next.unwind_steps >= base.unwind_steps);
            }
        }
    }
}

#[test]
fn insufficient_memo_is_atomic() {
    let links = [Link::Next(1), Link::Terminal];
    let mut memo = [MemoState::Done(Ok(77))];
    let mut path = [91, 92, 93];
    assert!(matches!(
        Resolver::new(&links, &mut memo, &mut path),
        Err(ScratchTooSmall)
    ));
    assert_eq!(memo, [MemoState::Done(Ok(77))]);
    assert_eq!(path, [91, 92, 93]);
}

#[test]
fn insufficient_path_is_atomic_even_with_sufficient_memo() {
    let links = [Link::Next(1), Link::Terminal];
    let mut memo = [MemoState::Visiting; 3];
    let mut path = [91];
    assert!(matches!(
        Resolver::new(&links, &mut memo, &mut path),
        Err(ScratchTooSmall)
    ));
    assert_eq!(memo, [MemoState::Visiting; 3]);
    assert_eq!(path, [91]);
}

#[test]
fn both_insufficient_capacities_are_atomic() {
    let links = [Link::Terminal; 2];
    let mut memo = [MemoState::Done(Err(ResolveError::Cycle))];
    let mut path = [];
    assert!(matches!(
        Resolver::new(&links, &mut memo, &mut path),
        Err(ScratchTooSmall)
    ));
    assert_eq!(memo, [MemoState::Done(Err(ResolveError::Cycle))]);
    assert!(path.is_empty());
}

#[test]
fn constructor_resets_only_live_memo_and_queries_preserve_spare_tails() {
    let links = [Link::Next(1), Link::Terminal];
    let mut memo = [MemoState::Done(Err(ResolveError::Rejected)); 4];
    let mut path = [71, 72, 73, 74];
    {
        let mut resolver = Resolver::new(&links, &mut memo, &mut path).unwrap();
        assert_eq!(resolver.memo, [MemoState::Unseen; 2]);
        assert_eq!(resolver.path, [71, 72]);
        assert_eq!(resolver.resolve(0), Ok(1));
        assert_eq!((resolver.visits, resolver.unwinds), (2, 2));
    }
    assert_eq!(&memo[2..], &[MemoState::Done(Err(ResolveError::Rejected)); 2]);
    assert_eq!(&path[2..], &[73, 74]);
}

#[test]
fn empty_graph_accepts_empty_scratch_and_rejects_all_starts() {
    let mut memo = [];
    let mut path = [];
    let mut resolver = Resolver::new(&[], &mut memo, &mut path).unwrap();
    assert_eq!(resolver.resolve(0), Err(ResolveError::Missing));
    assert_eq!(resolver.resolve(usize::MAX), Err(ResolveError::Missing));
    assert_eq!((resolver.visits, resolver.unwinds), (0, 0));
}

#[test]
fn empty_graph_preserves_all_spare_storage() {
    let mut memo = [MemoState::Visiting];
    let mut path = [99];
    {
        let mut resolver = Resolver::new(&[], &mut memo, &mut path).unwrap();
        assert_eq!(resolver.resolve(0), Err(ResolveError::Missing));
    }
    assert_eq!(memo, [MemoState::Visiting]);
    assert_eq!(path, [99]);
}

#[test]
fn invalid_starts_do_not_read_or_mutate_cache_or_path() {
    let links = [Link::Terminal];
    let mut memo = [MemoState::Visiting];
    let mut path = [99];
    let mut resolver = Resolver::new(&links, &mut memo, &mut path).unwrap();
    for start in [1, 2, usize::MAX] {
        assert_eq!(resolver.resolve(start), Err(ResolveError::Missing));
    }
    assert_eq!(resolver.memo, [MemoState::Unseen]);
    assert_eq!(resolver.path, [99]);
    assert_eq!((resolver.visits, resolver.unwinds), (0, 0));
    assert_eq!(resolver.resolve(0), Ok(0));
    assert_eq!(resolver.resolve(usize::MAX), Err(ResolveError::Missing));
    assert_eq!(resolver.memo, [MemoState::Done(Ok(0))]);
    assert_eq!((resolver.visits, resolver.unwinds), (1, 1));
}

#[test]
fn terminal_returns_its_own_index_and_repeated_queries_do_not_revisit() {
    let links = [Link::Terminal; 3];
    let mut memo = [MemoState::Visiting; 3];
    let mut path = [usize::MAX; 3];
    let mut resolver = Resolver::new(&links, &mut memo, &mut path).unwrap();
    for _ in 0..4 {
        for start in 0..3 {
            assert_eq!(resolver.resolve(start), Ok(start));
        }
    }
    assert_eq!((resolver.visits, resolver.unwinds), (3, 3));
}

#[test]
fn long_shared_suffix_has_one_visit_and_unwind_per_node() {
    const N: usize = 128;
    let mut links = [Link::Terminal; N];
    links[0] = Link::Next(2);
    links[1] = Link::Next(2);
    for (node, link) in links.iter_mut().enumerate().take(N - 1).skip(2) {
        *link = Link::Next(node + 1);
    }
    let mut memo = [MemoState::Unseen; N];
    let mut path = [usize::MAX; N];
    let mut resolver = Resolver::new(&links, &mut memo, &mut path).unwrap();
    assert_eq!(resolver.resolve(0), Ok(N - 1));
    assert_eq!((resolver.visits, resolver.unwinds), (N - 1, N - 1));
    assert_eq!(resolver.resolve(1), Ok(N - 1));
    assert_eq!((resolver.visits, resolver.unwinds), (N, N));
    for _ in 0..3 {
        for start in (0..N).rev() {
            assert_eq!(resolver.resolve(start), Ok(N - 1));
        }
    }
    assert_eq!((resolver.visits, resolver.unwinds), (N, N));
    assert!(resolver
        .memo
        .iter()
        .all(|state| *state == MemoState::Done(Ok(N - 1))));
}

#[test]
fn missing_edge_caches_valid_prefix_only() {
    let links = [Link::Next(1), Link::Next(usize::MAX), Link::Terminal];
    let mut memo = [MemoState::Unseen; 3];
    let mut path = [99; 3];
    let mut resolver = Resolver::new(&links, &mut memo, &mut path).unwrap();
    assert_eq!(resolver.resolve(0), Err(ResolveError::Missing));
    assert_eq!(
        resolver.memo,
        [
            MemoState::Done(Err(ResolveError::Missing)),
            MemoState::Done(Err(ResolveError::Missing)),
            MemoState::Unseen,
        ]
    );
    assert_eq!(resolver.resolve(1), Err(ResolveError::Missing));
    assert_eq!((resolver.visits, resolver.unwinds), (2, 2));
    assert_eq!(resolver.resolve(2), Ok(2));
}

#[test]
fn rejected_link_caches_prefix_and_keeps_other_component_independent() {
    let links = [Link::Next(1), Link::Reject, Link::Terminal];
    let mut memo = [MemoState::Unseen; 3];
    let mut path = [99; 3];
    let mut resolver = Resolver::new(&links, &mut memo, &mut path).unwrap();
    assert_eq!(resolver.resolve(0), Err(ResolveError::Rejected));
    assert_eq!(resolver.resolve(1), Err(ResolveError::Rejected));
    assert_eq!(resolver.memo[2], MemoState::Unseen);
    assert_eq!((resolver.visits, resolver.unwinds), (2, 2));
    assert_eq!(resolver.resolve(2), Ok(2));
    assert_eq!((resolver.visits, resolver.unwinds), (3, 3));
}

#[test]
fn self_cycle_and_prefix_are_fully_unwound() {
    let links = [Link::Next(1), Link::Next(1), Link::Next(0)];
    let mut memo = [MemoState::Unseen; 3];
    let mut path = [99; 3];
    let mut resolver = Resolver::new(&links, &mut memo, &mut path).unwrap();
    assert_eq!(resolver.resolve(0), Err(ResolveError::Cycle));
    assert_eq!((resolver.visits, resolver.unwinds), (2, 2));
    assert_eq!(resolver.resolve(1), Err(ResolveError::Cycle));
    assert_eq!(resolver.resolve(2), Err(ResolveError::Cycle));
    assert_eq!(resolver.memo, [MemoState::Done(Err(ResolveError::Cycle)); 3]);
    assert_eq!((resolver.visits, resolver.unwinds), (3, 3));
}

#[test]
fn multinode_cycle_and_incoming_prefix_are_fully_unwound() {
    let links = [Link::Next(1), Link::Next(2), Link::Next(3), Link::Next(1)];
    let mut memo = [MemoState::Unseen; 4];
    let mut path = [99; 4];
    let mut resolver = Resolver::new(&links, &mut memo, &mut path).unwrap();
    for start in [0, 3, 2, 1, 0] {
        assert_eq!(resolver.resolve(start), Err(ResolveError::Cycle));
        assert_eq!(resolver.memo, [MemoState::Done(Err(ResolveError::Cycle)); 4]);
        assert_eq!((resolver.visits, resolver.unwinds), (4, 4));
    }
}

#[test]
fn shared_failure_suffixes_have_linear_visit_and_unwind_counts() {
    for (last, expected) in [
        (Link::Reject, ResolveError::Rejected),
        (Link::Next(usize::MAX), ResolveError::Missing),
        (Link::Next(2), ResolveError::Cycle),
    ] {
        const N: usize = 64;
        let mut links = [last; N];
        links[0] = Link::Next(2);
        links[1] = Link::Next(2);
        for (node, link) in links.iter_mut().enumerate().take(N - 1).skip(2) {
            *link = Link::Next(node + 1);
        }
        let mut memo = [MemoState::Unseen; N];
        let mut path = [usize::MAX; N];
        let mut resolver = Resolver::new(&links, &mut memo, &mut path).unwrap();
        assert_eq!(resolver.resolve(0), Err(expected));
        assert_eq!((resolver.visits, resolver.unwinds), (N - 1, N - 1));
        assert_eq!(resolver.resolve(1), Err(expected));
        for _ in 0..3 {
            for start in 0..N {
                assert_eq!(resolver.resolve(start), Err(expected));
            }
        }
        assert_eq!((resolver.visits, resolver.unwinds), (N, N));
        assert!(resolver
            .memo
            .iter()
            .all(|state| *state == MemoState::Done(Err(expected))));
    }
}

// Independent oracle: no memo, no resolver helper, no cycle marks. A walk that
// survives n successor steps without terminating or leaving the graph repeats
// a node by the pigeonhole principle, so its failure category is Cycle.
fn bounded_reference(links: &[Link], start: usize) -> Result<usize, ResolveError> {
    let mut node = start;
    for _ in 0..=links.len() {
        match links.get(node) {
            None => return Err(ResolveError::Missing),
            Some(Link::Terminal) => return Ok(node),
            Some(Link::Reject) => return Err(ResolveError::Rejected),
            Some(Link::Next(next)) => node = *next,
        }
    }
    Err(ResolveError::Cycle)
}

#[test]
fn bounded_generated_graphs_match_independent_traversal_in_both_query_orders() {
    // Exhaust every graph of up to three nodes: terminal, reject, every valid
    // successor, and one invalid successor per node. Just 246 graphs total.
    for n in 0_usize..=3 {
        let radix = n + 3;
        let graph_count = radix.pow(n as u32);
        for encoding in 0..graph_count {
            let mut digits = encoding;
            let mut links = [Link::Terminal; 3];
            for link in &mut links[..n] {
                *link = match digits % radix {
                    0 => Link::Terminal,
                    1 => Link::Reject,
                    digit => Link::Next(digit - 2),
                };
                digits /= radix;
            }
            let links = &links[..n];
            for reverse in [false, true] {
                let mut memo = [MemoState::Visiting; 3];
                let mut path = [usize::MAX; 3];
                let mut resolver = Resolver::new(links, &mut memo, &mut path).unwrap();
                for _ in 0..2 {
                    for offset in 0..n {
                        let start = if reverse { n - 1 - offset } else { offset };
                        assert_eq!(
                            resolver.resolve(start),
                            bounded_reference(links, start),
                            "n={n}, encoding={encoding}, start={start}, reverse={reverse}"
                        );
                        assert!(!resolver.memo.contains(&MemoState::Visiting));
                        assert_eq!(resolver.visits, resolver.unwinds);
                        assert!(resolver.visits <= n);
                    }
                }
                assert_eq!(resolver.visits, n);
                assert_eq!(resolver.unwinds, n);
                let before = resolver.visits;
                for start in [n, usize::MAX] {
                    assert_eq!(resolver.resolve(start), bounded_reference(links, start));
                }
                assert_eq!(resolver.visits, before);
            }
        }
    }
}
