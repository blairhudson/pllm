use super::*;

// Independent public-ABI/event reconstruction. No production pools, row builders
// or closed-form constants are read by this oracle; u128 exposes overflow.
type EventRow = (u128, u128, u128, u128); // call pool, control pool, events, instances

fn abi<T>() -> u128 {
    std::mem::size_of::<T>() as u128
}

fn independent_rows(m: usize, a: usize) -> [EventRow; 6] {
    type P = (String, Vec<usize>);
    let u = abi::<usize>();
    let b = abi::<bool>();
    let unit = abi::<()>();
    let child = 3 * u + abi::<*mut u8>() + abi::<std::alloc::Layout>() + b;
    [
        (2 * abi::<&mut Vec<P>>() + 2 * unit,
            abi::<&mut [P]>() + 3 * u + b, 8, 1),
        (2 * abi::<(&mut [P], usize)>() + 2 * abi::<Option<&mut P>>(),
            u + b, 2, m as u128 + 1),
        (2 * abi::<&mut P>() + 2 * unit, u + abi::<u8>() + 2 * b, 5, m as u128),
        (2 * abi::<&mut String>() + 2 * unit, child, 6, m as u128),
        (2 * abi::<&mut Vec<usize>>() + 2 * unit, child, 6, m as u128),
        (2 * abi::<(*mut u8, std::alloc::Layout)>() + 2 * unit,
            u + b + unit, 3, a as u128),
    ]
}

fn independent_normalize(rows: &[EventRow], headers: u128) -> [u128; 5] {
    let (mut k, mut v, mut p, mut s) = (0, 0, headers, 0);
    for &(call, control, events, n) in rows {
        k += n;
        v += events * n;
        p += 2 * call * n + 2 * control * (events + 1) * n;
        if n != 0 {
            s += call + control;
        }
    }
    [k, v, p, s, k + v + p]
}

fn independent_sequence(m: usize, a: usize) -> [u128; 5] {
    independent_normalize(&independent_rows(m, a),
        abi::<Vec<(String, Vec<usize>)>>() + m as u128 * (abi::<String>() + abi::<Vec<usize>>()))
}

fn numbers(work: LogicalRetirementWork) -> [u128; 5] {
    [work.k as u128, work.v as u128, work.p as u128, work.s as u128,
        work.checked_total().unwrap() as u128]
}

fn dominates(upper: LogicalRetirementWork, lower: LogicalRetirementWork) {
    for (upper, lower) in numbers(upper).into_iter().zip(numbers(lower)) {
        assert!(upper >= lower);
    }
}

fn fixtures() -> Vec<Vec<(String, Vec<usize>)>> {
    let mut unallocated_children = Vec::with_capacity(4);
    unallocated_children.push((String::new(), Vec::new()));
    let mut allocated_empty = Vec::with_capacity(5);
    allocated_empty.push((String::with_capacity(17), Vec::with_capacity(9)));
    let mut mixed = Vec::with_capacity(12);
    mixed.push((String::from("rank-one"), vec![2]));
    mixed.push((String::new(), vec![3, 4]));
    mixed.push((String::with_capacity(31), Vec::with_capacity(7)));
    mixed.push((String::from("rank-many"), vec![1, 2, 3, 4, 5]));
    vec![Vec::new(), Vec::with_capacity(8), unallocated_children, allocated_empty, mixed]
}

fn independent_census(value: &Vec<(String, Vec<usize>)>) -> (usize, usize) {
    let mut releases = usize::from(value.capacity() != 0);
    let mut bytes = std::mem::size_of_val(value)
        + value.capacity() * std::mem::size_of::<(String, Vec<usize>)>();
    for (name, dims) in value {
        releases += usize::from(name.capacity() != 0) + usize::from(dims.capacity() != 0);
        bytes += name.capacity() + dims.capacity() * std::mem::size_of::<usize>();
    }
    (releases, bytes)
}

#[test]
fn independent_abi_event_rows_and_closed_algebra_agree() {
    for m in [0, 1, 2, 3, 7, 19, 63] {
        for a in 0..=2 * m + 1 {
            let actual = sequence_work(m, a).unwrap();
            assert_eq!(numbers(actual), independent_sequence(m, a));
            assert_eq!(actual.k, (2 + 4 * m + a) as u64);
            assert_eq!(actual.v, (10 + 19 * m + 3 * a) as u64);
        }
    }
}

#[test]
fn concrete_empty_spare_allocated_and_mixed_rank_census() {
    for value in fixtures() {
        let (a, h) = independent_census(&value);
        let quote = string_usize_sequence_retirement_v1(&value).unwrap();
        assert_eq!(quote.initialized_pairs, value.len());
        assert_eq!(quote.allocation_releases, a);
        assert_eq!(quote.h_owner, h);
        assert_eq!(numbers(quote.work), independent_sequence(value.len(), a));
        dominates(string_usize_sequence_retirement_bound_v1(value.len()).unwrap(), quote.work);
    }
    let empty = string_usize_sequence_retirement_v1(&Vec::new()).unwrap();
    let reserved = string_usize_sequence_retirement_v1(&Vec::with_capacity(3)).unwrap();
    assert_eq!((empty.initialized_pairs, empty.allocation_releases), (0, 0));
    assert_eq!((reserved.initialized_pairs, reserved.allocation_releases), (0, 1));
    assert!(empty.work.k > 0 && empty.work.v > 0 && empty.work.s > 0);
}

#[test]
fn standalone_children_reconstruct_zero_and_nonzero_release_rows() {
    for name in [String::new(), String::with_capacity(31), String::from("payload")] {
        let quote = string_retirement_v1(&name).unwrap();
        let a = usize::from(name.capacity() != 0);
        let rows = independent_rows(1, a);
        assert_eq!(numbers(quote.work), independent_normalize(&[rows[3], rows[5]], abi::<String>()));
        assert_eq!(quote.initialized_pairs, 0);
        assert_eq!(quote.allocation_releases, a);
        assert_eq!(quote.h_owner, std::mem::size_of::<String>() + name.capacity());
    }
    for dims in [Vec::new(), Vec::with_capacity(19), vec![1, 2, 3, 4]] {
        let quote = usize_vec_retirement_v1(&dims).unwrap();
        let a = usize::from(dims.capacity() != 0);
        let rows = independent_rows(1, a);
        assert_eq!(numbers(quote.work), independent_normalize(&[rows[4], rows[5]], abi::<Vec<usize>>()));
        assert_eq!(quote.initialized_pairs, 0);
        assert_eq!(quote.allocation_releases, a);
        assert_eq!(quote.h_owner, std::mem::size_of::<Vec<usize>>() + dims.capacity() * std::mem::size_of::<usize>());
    }
}

#[test]
fn capacity_changes_h_not_work_and_payload_lengths_add_no_visits() {
    let mut value = Vec::with_capacity(4);
    value.push((String::with_capacity(8), Vec::with_capacity(8)));
    let initial = string_usize_sequence_retirement_v1(&value).unwrap();
    value[0].0.push_str("abc");
    value[0].1.extend_from_slice(&[usize::MAX, 0, 7]);
    assert_eq!(string_usize_sequence_retirement_v1(&value).unwrap(), initial);
    let old_name = value[0].0.capacity();
    let old_dims = value[0].1.capacity();
    let old_outer = value.capacity();
    value[0].0.reserve_exact(old_name + 17);
    value[0].1.reserve_exact(old_dims + 13);
    value.reserve_exact(old_outer + 11);
    assert!(value[0].0.capacity() > old_name);
    assert!(value[0].1.capacity() > old_dims);
    assert!(value.capacity() > old_outer);
    let grown = string_usize_sequence_retirement_v1(&value).unwrap();
    assert_eq!(grown.work, initial.work);
    assert_eq!(grown.allocation_releases, initial.allocation_releases);
    assert!(grown.h_owner > initial.h_owner);
    assert_eq!(grown.h_owner, independent_census(&value).1);
}

#[test]
fn allocation_branch_adds_exactly_one_release_row() {
    let mut value = vec![(String::new(), Vec::new())];
    let before = string_usize_sequence_retirement_v1(&value).unwrap();
    value[0].0.reserve_exact(29);
    let after = string_usize_sequence_retirement_v1(&value).unwrap();
    let release = independent_normalize(&[independent_rows(0, 1)[5]], 0);
    assert_eq!(after.allocation_releases, before.allocation_releases + 1);
    let x = numbers(after.work);
    let y = numbers(before.work);
    for i in [0, 1, 2, 4] {
        assert_eq!(x[i] - y[i], release[i]);
    }
    // Outer allocation already made the reusable release pool live.
    assert_eq!(after.work.s, before.work.s);
}

#[test]
fn scalar_bound_dominates_all_partial_counts_without_capacity_claim() {
    for maximum in 0..=25 {
        let bound = string_usize_sequence_retirement_bound_v1(maximum).unwrap();
        assert_eq!(numbers(bound), independent_sequence(maximum, 2 * maximum + 1));
        for initialized in 0..=maximum {
            for releases in 0..=2 * initialized + 1 {
                dominates(bound, sequence_work(initialized, releases).unwrap());
            }
        }
    }
    let exact_empty = string_usize_sequence_retirement_v1(&Vec::new()).unwrap();
    assert!(string_usize_sequence_retirement_bound_v1(0).unwrap().k > exact_empty.work.k);
}

#[test]
fn checked_overflow_matches_wide_algebra_and_private_zst_eligibility() {
    assert_eq!(allocation_facts(0, 7), Ok((0, 0)));
    assert_eq!(allocation_facts(usize::MAX, 0), Ok((0, 0)));
    assert_eq!(allocation_facts(0, 0), Ok((0, 0)));
    assert_eq!(allocation_facts(3, 7), Ok((21, 1)));
    assert_eq!(allocation_facts(usize::MAX, 2), Err(Error::Overflow));
    assert_eq!(checked_array_bytes::<usize>(usize::MAX), Err(Error::Overflow));
    assert_eq!(bytes_add(usize::MAX, 1), Err(Error::Overflow));
    assert_eq!(add(u64::MAX, 1), Err(Error::Overflow));
    assert_eq!(mul(u64::MAX, 2), Err(Error::Overflow));
    assert_eq!(LogicalRetirementWork { k: u64::MAX, v: 1, p: 0, s: 0 }.checked_total(), Err(Error::Overflow));
    for m in [0, 1, usize::MAX / 100_000, usize::MAX / 128, usize::MAX / 2, usize::MAX] {
        for a in [0, 1, usize::MAX / 2, usize::MAX] {
            let expected = independent_sequence(m, a);
            let fits = m.checked_add(1).is_some()
                && expected.into_iter().all(|n| n <= u64::MAX as u128);
            match sequence_work(m, a) {
                Ok(work) => { assert!(fits); assert_eq!(numbers(work), expected); }
                Err(error) => { assert!(!fits); assert_eq!(error, Error::Overflow); }
            }
        }
    }
    assert_eq!(string_usize_sequence_retirement_bound_v1(usize::MAX), Err(Error::Overflow));
    assert_eq!(string_usize_sequence_retirement_bound_v1(usize::MAX / 2 + 1), Err(Error::Overflow));
    // Above release-count arithmetic can fit while normalized byte/work cannot.
    if usize::BITS >= 64 {
        assert_eq!(string_usize_sequence_retirement_bound_v1(usize::MAX / 4), Err(Error::Overflow));
    }
}

#[test]
fn query_and_unused_numeric_quote_leave_real_owners_unchanged() {
    fn consume<T>(value: T) { drop(value); }
    for value in fixtures() {
        let outer = (value.as_ptr(), value.len(), value.capacity());
        let children: Vec<_> = value.iter().map(|(s, v)|
            (s.as_ptr(), s.len(), s.capacity(), v.as_ptr(), v.len(), v.capacity(), s.clone(), v.clone())).collect();
        let quote = string_usize_sequence_retirement_v1(&value).unwrap();
        consume(quote);
        assert_eq!((value.as_ptr(), value.len(), value.capacity()), outer);
        for ((s, v), before) in value.iter().zip(children) {
            consume(string_retirement_v1(s).unwrap());
            consume(usize_vec_retirement_v1(v).unwrap());
            assert_eq!((s.as_ptr(), s.len(), s.capacity(), v.as_ptr(), v.len(), v.capacity()),
                (before.0, before.1, before.2, before.3, before.4, before.5));
            assert_eq!(s, &before.6);
            assert_eq!(v, &before.7);
        }
    }
}

#[test]
fn independent_child_masks_partition_prefix_and_cleanup_once() {
    // Enumerate logical completed-child masks independently of std's drop order.
    // Alternate eligibility, including an empty-but-allocated conceptual child.
    for pairs in [2usize, 3] {
        let eligible: Vec<[bool; 2]> = (0..pairs).map(|i| [true, i % 2 == 0]).collect();
        let releases = 1 + eligible.iter().flatten().filter(|&&x| x).count();
        let rows = independent_rows(pairs, releases);
        let whole = independent_sequence(pairs, releases);
        for encoded in 0..4usize.pow(pairs as u32) {
            let mut masks = encoded;
            let (mut done_children, mut done_pairs, mut started_pairs, mut done_releases) = (0, 0, 0, 0);
            let mut pair_prefix_events = 0;
            for allocation in &eligible {
                let mask = masks % 4;
                masks /= 4;
                if mask != 0 { started_pairs += 1; }
                if mask == 3 { done_pairs += 1; }
                let completed = mask.count_ones() as usize;
                done_children += completed;
                pair_prefix_events += match mask { 0 => 0, 3 => 5, _ => 2 + completed };
                for (child, &allocated) in allocation.iter().enumerate() {
                    if mask & (1 << child) != 0 && allocated { done_releases += 1; }
                }
            }
            // Outer release and terminal remain in suffix. No second driver,
            // no duplicated pair call for a pair whose first child completed.
            let prefix_calls = 1 + done_pairs + started_pairs + done_children + done_releases;
            let suffix_calls = (pairs + 1 - done_pairs) + (pairs - started_pairs)
                + (2 * pairs - done_children) + (releases - done_releases);
            assert_eq!(prefix_calls + suffix_calls, whole[0] as usize);
            let prefix_visits = 5 + 2 * done_pairs + pair_prefix_events + 6 * done_children + 3 * done_releases;
            let suffix_visits = 3 + 2 * (pairs + 1 - done_pairs) + (5 * pairs - pair_prefix_events)
                + 6 * (2 * pairs - done_children) + 3 * (releases - done_releases);
            assert_eq!(prefix_visits + suffix_visits, whole[1] as usize);
            // Partition each row's potential-producing events/instances too.
            // Child pools have identical ABI, but distinct reusable S slots.
            let row_prefix = [1, done_pairs, started_pairs, 0, 0, done_releases];
            let mut prefix_p = 0u128;
            let mut suffix_p = 0u128;
            for (index, &(call, control, events, instances)) in rows[..6].iter().enumerate() {
                let (calls, visits) = match index {
                    0 => (1, 5),
                    2 => (started_pairs, pair_prefix_events),
                    3 | 4 => (0, 0),
                    _ => (row_prefix[index], row_prefix[index] * events as usize),
                };
                let (calls, visits) = if index == 3 || index == 4 {
                    let child_index = index - 3;
                    let completed = (0..pairs).filter(|i| (encoded >> (2 * i)) & (1 << child_index) != 0).count();
                    (completed, 6 * completed)
                } else { (calls, visits) };
                // Ctl's +1 entry/retirement slot is assigned once per instance.
                let prefix = 2 * call * calls as u128 + 2 * control * (visits + calls) as u128;
                let full = 2 * call * instances + 2 * control * (events + 1) * instances;
                assert!(prefix <= full);
                prefix_p += prefix;
                suffix_p += full - prefix;
            }
            let headers = abi::<Vec<(String, Vec<usize>)>>()
                + pairs as u128 * (abi::<String>() + abi::<Vec<usize>>());
            suffix_p += headers; // Conservatively leave every header read pending.
            assert_eq!(prefix_p + suffix_p, whole[2]);
            let remaining = suffix_calls as u128 + suffix_visits as u128 + suffix_p;
            assert!(remaining <= whole[4]);
            assert!(whole[3] <= numbers(string_usize_sequence_retirement_bound_v1(pairs).unwrap())[3]);
        }
    }
}
