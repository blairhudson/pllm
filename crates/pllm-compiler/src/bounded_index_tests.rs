use super::keyed::{Entry, Key, KeyStats, Quote, Scalar, Strings};
use super::*;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::rc::Rc;

fn reset_factories() {
    FACTORIES.with(|count| count.set(0));
}

fn factories() -> usize {
    FACTORIES.with(Cell::get)
}

#[test]
fn exact_box_preserves_initializer_order_values_and_factory_count() {
    for count in [0, 1, 5, 32] {
        reset_factories();
        let mut visits = Vec::new();
        let rows = exact_box(count, |index| {
            visits.push(index);
            (index, count - index)
        })
        .unwrap();
        assert_eq!(visits, (0..count).collect::<Vec<_>>());
        assert_eq!(rows.len(), count);
        let expected: Vec<_> = (0..count).map(|index| (index, count - index)).collect();
        assert_eq!(rows.as_ref(), expected.as_slice());
        assert_eq!(factories(), 1);
    }
    reset_factories();
    let mut visited = false;
    assert!(matches!(
        exact_box::<u64>(usize::MAX, |_| {
            visited = true;
            0
        }),
        Err(Error::Overflow)
    ));
    assert!(!visited);
    assert_eq!(factories(), 0);
}

#[test]
fn exact_box_initializer_panic_stops_at_original_ordinal() {
    reset_factories();
    let mut visits = Vec::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = exact_box(5, |index| {
            visits.push(index);
            assert_ne!(index, 2, "initializer failure");
            index
        });
    }));
    assert!(result.is_err());
    assert_eq!(visits, [0, 1, 2]);
    assert_eq!(factories(), 1);
}

#[test]
fn shared_buffer_errors_keep_compiler_mapping() {
    assert_eq!(buffer_error(backing::Error::Overflow), Error::Overflow);
    assert_eq!(buffer_error(backing::Error::Allocation), Error::Capacity);
    assert_eq!(buffer_error(backing::Error::Incomplete), Error::Incomplete);
}

// Frozen pre-extraction runtime driver. Compare traces as well as output/work;
// equal keys may permute under heapsort, so sorted-value parity alone is weaker.
fn original_grouping_sort<T>(
    items: &mut [T],
    usage: &mut Usage,
    mut compare: impl FnMut(&T, &T, &mut Usage) -> Ordering,
) {
    fn sift<T>(
        items: &mut [T],
        root: &mut usize,
        end: usize,
        usage: &mut Usage,
        compare: &mut impl FnMut(&T, &T, &mut Usage) -> Ordering,
    ) {
        while *root < end / 2 {
            let mut child = 2 * *root + 1;
            if child + 1 < end {
                usage.comparisons += 1;
                usage.work += 1;
                if compare(&items[child], &items[child + 1], usage).is_lt() {
                    child += 1;
                }
            }
            usage.comparisons += 1;
            usage.work += 1;
            if !compare(&items[*root], &items[child], usage).is_lt() {
                break;
            }
            items.swap(*root, child);
            usage.work += 1;
            *root = child;
        }
    }
    let mut root = items.len() / 2;
    let mut end = items.len();
    while root > 0 {
        let next = root - 1;
        root = next;
        sift(items, &mut root, end, usage, &mut compare);
        root = next;
    }
    while end > 1 {
        end -= 1;
        items.swap(0, end);
        usage.work += 1;
        root = 0;
        sift(items, &mut root, end, usage, &mut compare);
    }
}

#[test]
fn heapsort_matches_original_trace_meter_and_boundary_lengths() {
    for power in 0..=8 {
        let n = 1usize << power;
        for len in [n - 1, n, n + 1] {
            for mode in 0..4 {
                let input: Vec<_> = (0..len)
                    .map(|i| {
                        let key = match mode {
                            0 => i,
                            1 => len - i,
                            2 => 0,
                            _ => (i * 37) % 17,
                        };
                        (key, i)
                    })
                    .collect();
                let mut actual = input.clone();
                let mut expected = input;
                let mut actual_usage = Usage::default();
                let mut expected_usage = Usage::default();
                let mut actual_trace = Vec::new();
                let mut expected_trace = Vec::new();
                heapsort(&mut actual, &mut actual_usage, |a, b, usage| {
                    usage.step();
                    actual_trace.push((a.1, b.1));
                    a.0.cmp(&b.0)
                });
                original_grouping_sort(&mut expected, &mut expected_usage, |a, b, usage| {
                    usage.work += 1;
                    expected_trace.push((a.1, b.1));
                    a.0.cmp(&b.0)
                });
                assert_eq!(actual, expected, "len={len}, mode={mode}");
                assert_eq!(actual_trace, expected_trace);
                assert_eq!(actual_usage, expected_usage);
                assert!(actual.windows(2).all(|pair| pair[0].0 <= pair[1].0));
                let bound = sort_comparison_bound(len).unwrap();
                let original_bound = if len < 2 {
                    0
                } else {
                    2 * ((len / 2 + len - 1) as u128) * (usize::BITS - len.leading_zeros()) as u128
                };
                assert_eq!(bound, original_bound);
                assert!(actual_usage.comparisons <= bound);
                assert!(actual_usage.work <= 3 * bound + len.saturating_sub(1) as u128);
            }
        }
    }
}

#[test]
fn shared_lower_bound_preserves_original_probe_order() {
    for len in [0, 1, 2, 3, 7, 8, 9, 31, 32, 33, 1024] {
        let input: Vec<_> = (0..len).map(|i| i / 3).collect();
        for query in [0, 1, len / 6, len, usize::MAX] {
            let mut original_trace = Vec::new();
            let (mut lo, mut hi) = (0, input.len());
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                original_trace.push(input[mid]);
                if input[mid] < query {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let mut trace = Vec::new();
            let found = lower_bound_by(&input, |&value| {
                trace.push(value);
                value.cmp(&query)
            });
            assert_eq!(found, lo);
            assert_eq!(trace, original_trace);
            assert!(trace.len() as u128 <= search_comparison_bound(len));
        }
    }
}

#[test]
fn indexed_first_last_and_positions_match_original_tree_and_sorted_oracles() {
    let keys = ["z", "", "é", "z", "prefix", "", "prefix.long", "λ", "z"];
    let entries: Vec<_> = keys
        .iter()
        .enumerate()
        .map(|(position, &key)| Entry {
            key,
            position,
            value: position,
        })
        .collect();
    let mut first = BTreeMap::new();
    let mut last = BTreeMap::new();
    for entry in &entries {
        first.entry(entry.key).or_insert(entry.position);
        last.insert(entry.key, entry.position);
    }
    let quote = Quote::checked(&entries, u128::MAX).unwrap();
    let resources = quote.resources();
    let index = quote
        .construct(resources.workspace_bytes, resources.construction_work)
        .unwrap();
    for key in keys.into_iter().chain(["missing", "prefix.l", "💡"]) {
        let mut usage = Usage::default();
        let matches = index
            .matches(&key, &mut usage, resources.lookup_work)
            .unwrap();
        assert_eq!(
            matches.first().map(|entry| entry.position),
            first.get(key).copied()
        );
        assert_eq!(
            matches.last().map(|entry| entry.position),
            last.get(key).copied()
        );
        let expected: Vec<_> = entries.iter().filter(|entry| entry.key == key).collect();
        assert_eq!(matches.len(), expected.len());
        for (actual, expected) in matches.iter().zip(expected) {
            assert!(std::ptr::eq(actual, expected));
        }
        assert!(usage.work <= resources.lookup_work);
        assert!(usage.comparisons <= resources.lookup_comparisons);
    }
}

#[test]
fn duplicate_position_ties_forward_references_and_predecessors_are_explicit() {
    // Deliberately not chronological slice order. Equal positions use original
    // ordinal; neither first/last nor validation precedence relies on sort stability.
    let entries = [
        Entry {
            key: "later",
            position: 9,
            value: "future",
        },
        Entry {
            key: "same",
            position: 7,
            value: "late",
        },
        Entry {
            key: "same",
            position: 2,
            value: "first",
        },
        Entry {
            key: "same",
            position: 2,
            value: "second",
        },
        Entry {
            key: "input",
            position: 0,
            value: "public",
        },
    ];
    let quote = Quote::checked(&entries, u128::MAX).unwrap();
    let resources = quote.resources();
    let index = quote
        .construct(resources.workspace_bytes, resources.construction_work)
        .unwrap();
    let mut usage = Usage::default();
    let same = index.matches(&"same", &mut usage, u128::MAX).unwrap();
    assert_eq!(
        same.iter().map(|entry| entry.value).collect::<Vec<_>>(),
        ["first", "second", "late"]
    );
    assert_eq!(same.first().unwrap().value, "first");
    assert_eq!(same.last().unwrap().value, "late");
    assert!(!same.precedes(2, 2));
    assert!(same.precedes(2, 3));
    assert!(same.precedes(7, 1));
    assert!(same.first_before(2).is_none());
    assert_eq!(same.first_before(3).unwrap().value, "first");
    assert!(same
        .last_before(2, &mut usage, u128::MAX)
        .unwrap()
        .is_none());
    assert_eq!(
        same.last_before(3, &mut usage, u128::MAX)
            .unwrap()
            .unwrap()
            .value,
        "second"
    );
    assert_eq!(
        same.last_before(8, &mut usage, u128::MAX)
            .unwrap()
            .unwrap()
            .value,
        "late"
    );
    let future = index.matches(&"later", &mut usage, u128::MAX).unwrap();
    assert!(future.first_before(9).is_none());
    assert!(future.first_before(10).is_some());
    assert!(index
        .matches_for_input(3, &mut usage, u128::MAX)
        .unwrap()
        .precedes(2, 3));
    assert!(matches!(
        index.matches_for_input(99, &mut usage, u128::MAX),
        Err(Error::Position)
    ));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Direction {
    Input,
    Output,
}
impl super::keyed::ScalarValue for Direction {}

#[test]
fn scalar_option_and_directional_namespaces_keep_original_tuple_order() {
    let keys = [
        (Scalar(Direction::Output), Some(Scalar(0_u64)), "cache"),
        (Scalar(Direction::Input), Some(Scalar(1_u64)), "cache"),
        (Scalar(Direction::Input), None, "cache"),
        (Scalar(Direction::Input), Some(Scalar(0_u64)), "cache"),
        (Scalar(Direction::Input), Some(Scalar(0_u64)), "cache"),
        (Scalar(Direction::Input), Some(Scalar(0_u64)), "é"),
    ];
    let entries: Vec<_> = keys
        .iter()
        .enumerate()
        .map(|(position, &key)| Entry {
            key,
            position,
            value: position,
        })
        .collect();
    let quote = Quote::checked(&entries, u128::MAX).unwrap();
    let resources = quote.resources();
    let index = quote
        .construct(resources.workspace_bytes, resources.construction_work)
        .unwrap();
    let mut ordered: Vec<_> = (0..entries.len()).collect();
    ordered.sort_by_key(|&ordinal| (entries[ordinal].key, entries[ordinal].position, ordinal));
    assert_eq!(index.ordered_ordinals(), ordered);
    for key in &keys {
        let mut expected: Vec<_> = entries.iter().filter(|entry| entry.key == *key).collect();
        expected.sort_by_key(|entry| entry.position);
        let mut usage = Usage::default();
        let matches = index
            .matches(key, &mut usage, resources.lookup_work)
            .unwrap();
        assert_eq!(
            matches.iter().map(|entry| entry.value).collect::<Vec<_>>(),
            expected.iter().map(|entry| entry.value).collect::<Vec<_>>()
        );
    }
    assert!(index.usage().work <= resources.construction_work);
    assert!(index.usage().comparisons <= resources.construction_comparisons);
}

#[test]
fn ordered_string_lists_prefixes_empty_long_and_utf8_match_original_ord() {
    let long = "common-prefix-".repeat(64);
    let keys: Vec<Vec<String>> = vec![
        vec![],
        vec!["".into()],
        vec!["a".into()],
        vec!["a".into(), "".into()],
        vec!["a".into(), "b".into()],
        vec!["b".into(), "a".into()],
        vec!["é".into()],
        vec!["λ".into(), "💡".into()],
        vec![format!("{long}a"), "tail".into()],
        vec![format!("{long}b")],
        vec!["a".into(), "b".into()],
    ];
    let entries: Vec<_> = keys
        .iter()
        .rev()
        .enumerate()
        .map(|(position, key)| Entry {
            key: Strings(key),
            position,
            value: key,
        })
        .collect();
    let quote = Quote::checked(&entries, u128::MAX).unwrap();
    let resources = quote.resources();
    let index = quote
        .construct(resources.workspace_bytes, resources.construction_work)
        .unwrap();
    let mut expected: Vec<_> = entries.iter().enumerate().collect();
    expected.sort_by(|(ai, a), (bi, b)| {
        a.key
            .0
            .cmp(b.key.0)
            .then(a.position.cmp(&b.position))
            .then(ai.cmp(bi))
    });
    assert_eq!(
        index.ordered_ordinals(),
        expected.iter().map(|(i, _)| *i).collect::<Vec<_>>()
    );
    for key in &keys {
        let mut usage = Usage::default();
        let matches = index
            .matches(&Strings(key), &mut usage, resources.lookup_work)
            .unwrap();
        assert_eq!(
            matches.len(),
            entries.iter().filter(|entry| entry.key.0 == key).count()
        );
    }
    assert!(index.usage().work <= resources.construction_work);
}

#[test]
fn exact_workspace_and_work_one_short_precede_actual_factory() {
    let entries = [Entry {
        key: "original",
        position: 0,
        value: (),
    }];
    reset_factories();
    assert!(matches!(Quote::checked(&entries, 0), Err(Error::WorkLimit)));
    assert!(matches!(Quote::checked(&entries, 1), Err(Error::WorkLimit)));
    assert_eq!(factories(), 0);
    let resources = Quote::checked(&entries, 2).unwrap().resources();
    assert_eq!(resources.quote_work, 2);
    assert!(matches!(
        Quote::checked(&entries, 2).unwrap().construct(0, u128::MAX),
        Err(Error::Capacity)
    ));
    assert!(matches!(
        Quote::checked(&entries, 2)
            .unwrap()
            .construct(resources.workspace_bytes - 1, resources.construction_work),
        Err(Error::Capacity)
    ));
    assert!(matches!(
        Quote::checked(&entries, 2)
            .unwrap()
            .construct(resources.workspace_bytes, resources.construction_work - 1),
        Err(Error::WorkLimit)
    ));
    assert_eq!(factories(), 0);
    let index = Quote::checked(&entries, 2)
        .unwrap()
        .construct(resources.workspace_bytes, resources.construction_work)
        .unwrap();
    assert_eq!(factories(), 1);
    assert_eq!(
        resources.heap_bytes,
        std::mem::size_of_val(index.ordered_ordinals())
    );
    let mut usage = Usage::default();
    assert!(matches!(
        index.matches(&"original", &mut usage, resources.lookup_work - 1),
        Err(Error::WorkLimit)
    ));
    assert_eq!(usage, Usage::default());
    assert!(index
        .matches(&"original", &mut usage, resources.lookup_work)
        .unwrap()
        .first()
        .is_some());
    assert_eq!(factories(), 1); // Lookups allocate nothing.
}

#[test]
fn empty_index_and_lookup_counter_overflow_are_defined() {
    let entries: [Entry<&str, ()>; 0] = [];
    let quote = Quote::checked(&entries, 0).unwrap();
    let resources = quote.resources();
    assert_eq!(resources.heap_bytes, 0);
    assert_eq!(resources.construction_work, 0);
    assert_eq!(resources.lookup_work, 0);
    let index = quote.construct(resources.workspace_bytes, 0).unwrap();
    let mut usage = Usage::default();
    let matches = index.matches(&"anything", &mut usage, 0).unwrap();
    assert_eq!(matches.len(), 0);
    assert!(matches.is_empty());
    assert!(matches.first().is_none());
    assert!(matches.last().is_none());
    assert!(!matches.precedes(0, 0));
    assert_eq!(usage, Usage::default());

    let entries = [Entry {
        key: "x",
        position: 0,
        value: (),
    }];
    let quote = Quote::checked(&entries, u128::MAX).unwrap();
    let resources = quote.resources();
    let index = quote
        .construct(resources.workspace_bytes, resources.construction_work)
        .unwrap();
    for initial in [
        Usage {
            work: u128::MAX,
            comparisons: 0,
        },
        Usage {
            work: 0,
            comparisons: u128::MAX,
        },
    ] {
        let mut usage = initial;
        assert!(matches!(
            index.matches(&"x", &mut usage, u128::MAX),
            Err(Error::Overflow)
        ));
        assert_eq!(usage, initial);
    }
    assert_eq!(checked_add(u128::MAX, 1), Err(Error::Overflow));
    assert_eq!(checked_mul(u128::MAX, 2), Err(Error::Overflow));
    assert_eq!(layout_bytes::<u64>(usize::MAX), Err(Error::Overflow));
    let mut usage = Usage {
        work: u128::MAX,
        comparisons: 0,
    };
    assert_eq!(usage.charge(1, u128::MAX), Err(Error::Overflow));
    assert_eq!(usage.work, u128::MAX);
}

#[test]
fn index_growth_is_n_log_n_and_duplicate_lookup_is_logarithmic() {
    for duplicate in [false, true] {
        let mut previous_work = None;
        for n in [32, 64, 128, 256, 512, 1024, 2048] {
            let keys: Vec<_> = (0..n)
                .map(|i| format!("key.{:08}", if duplicate { 0 } else { i }))
                .collect();
            let entries: Vec<_> = keys
                .iter()
                .rev()
                .enumerate()
                .map(|(position, key)| Entry {
                    key: key.as_str(),
                    position,
                    value: position,
                })
                .collect();
            let quote = Quote::checked(&entries, (2 * n) as u128).unwrap();
            let resources = quote.resources();
            let index = quote
                .construct(resources.workspace_bytes, resources.construction_work)
                .unwrap();
            assert_eq!(index.resources(), resources);
            assert_eq!(resources.heap_bytes, layout_bytes::<usize>(n).unwrap());
            assert!(index.usage().work <= resources.construction_work);
            assert!(index.usage().comparisons <= resources.construction_comparisons);
            if let Some(previous) = previous_work {
                assert!(resources.construction_work < 3 * previous);
            }
            previous_work = Some(resources.construction_work);
            let mut usage = Usage::default();
            for key in &keys {
                let before = usage;
                let found = index.matches(&key.as_str(), &mut usage, u128::MAX).unwrap();
                assert_eq!(found.len(), if duplicate { n } else { 1 });
                assert!(usage.work - before.work <= resources.lookup_work);
                assert!(usage.comparisons - before.comparisons <= resources.lookup_comparisons);
            }
        }
    }
}

#[derive(Debug)]
struct DropProbe(Rc<Cell<usize>>);
impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[test]
fn original_key_and_value_owners_are_borrowed_not_cloned_or_replaced() {
    let mut key = String::with_capacity(4096);
    key.push_str("genuine-original");
    let original_capacity = key.capacity();
    let drops = Rc::new(Cell::new(0));
    let entries = [Entry {
        key: key.as_str(),
        position: 0,
        value: DropProbe(drops.clone()),
    }];
    let quote = Quote::checked(&entries, 2).unwrap();
    assert!(std::ptr::eq(quote.original_entries(), entries.as_slice()));
    let resources = quote.resources();
    let index = quote
        .construct(resources.workspace_bytes, resources.construction_work)
        .unwrap();
    let mut usage = Usage::default();
    let found = index
        .matches(&key.as_str(), &mut usage, resources.lookup_work)
        .unwrap();
    assert!(std::ptr::eq(found.first().unwrap(), &entries[0]));
    assert_eq!(found.first().unwrap().key.as_ptr(), key.as_ptr());
    assert_eq!(key.capacity(), original_capacity);
    assert_eq!(drops.get(), 0);
    drop(index);
    assert_eq!(drops.get(), 0);
    drop(entries);
    assert_eq!(drops.get(), 1);
}

#[test]
fn exact_prefix_storage_cleans_error_panic_and_transfer_paths() {
    let drops = Rc::new(Cell::new(0));
    let mut buffer = ExactBuffer::new(3).unwrap();
    buffer.push(DropProbe(drops.clone())).unwrap();
    buffer.push(DropProbe(drops.clone())).unwrap();
    assert!(matches!(buffer.finish(), Err(Error::Incomplete)));
    assert_eq!(drops.get(), 2);
    let mut buffer = ExactBuffer::new(1).unwrap();
    buffer.push(DropProbe(drops.clone())).unwrap();
    assert_eq!(buffer.push(DropProbe(drops.clone())), Err(Error::Capacity));
    assert_eq!(drops.get(), 3); // Rejected owned argument cleaned before return.
    let complete = buffer.finish().unwrap();
    drop(complete);
    assert_eq!(drops.get(), 4);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut buffer = ExactBuffer::new(4).unwrap();
        buffer.push(DropProbe(drops.clone())).unwrap();
        buffer.push(DropProbe(drops.clone())).unwrap();
        panic!("initializer failure");
    }));
    assert!(result.is_err());
    assert_eq!(drops.get(), 6);
    reset_factories();
    assert!(matches!(
        ExactBuffer::<u64>::new(usize::MAX),
        Err(Error::Overflow)
    ));
    assert_eq!(factories(), 0);
}

#[test]
fn key_prepass_budget_stops_before_each_nested_collection_visit() {
    let keys = vec!["one".into(), "two".into(), "three".into()];
    let entries = [Entry {
        key: Strings(&keys),
        position: 0,
        value: (),
    }];
    // One entry + one list + one visit per string; no byte-content scan needed.
    assert!(matches!(Quote::checked(&entries, 4), Err(Error::WorkLimit)));
    let quote = Quote::checked(&entries, 5).unwrap();
    assert_eq!(quote.resources().quote_work, 5);
    let mut usage = Usage::default();
    assert_eq!(Strings(&keys).quote_work(&mut usage, 4).unwrap(), 15);
    assert_eq!(usage.work, 4);
}

#[test]
fn duplicate_source_keys_share_one_original_ledger_cell_without_scanning() {
    let entries = [
        Entry {
            key: "shared",
            position: 0,
            value: Cell::new(None::<[u64; 2]>),
        },
        Entry {
            key: "other",
            position: 1,
            value: Cell::new(None),
        },
        Entry {
            key: "shared",
            position: 2,
            value: Cell::new(None),
        },
    ];
    let quote = Quote::checked(&entries, u128::MAX).unwrap();
    let resources = quote.resources();
    let index = quote
        .construct(resources.workspace_bytes, resources.construction_work)
        .unwrap();
    let mut usage = Usage::default();
    for ordinal in [0, 2] {
        let found = index
            .matches_for_input(ordinal, &mut usage, u128::MAX)
            .unwrap();
        let cell = &found.first().unwrap().value;
        assert!(std::ptr::eq(cell, &entries[0].value));
        match cell.get() {
            None => cell.set(Some([4, 8])),
            Some(previous) => assert_eq!(previous, [4, 8]),
        }
    }
    assert_eq!(entries[0].value.get(), Some([4, 8]));
    assert_eq!(entries[2].value.get(), None);
    assert!(usage.work <= 2 * resources.lookup_work);
}

#[test]
fn key_only_prequote_matches_actual_rows_before_any_derived_factory() {
    let keys = ["original", "", "é", "original", "prefix.long"];
    reset_factories();
    let mut stats = KeyStats::<&str, usize>::new();
    for key in keys {
        stats.push(key, 10).unwrap();
    }
    assert_eq!(stats.count(), keys.len());
    let prequote = stats.resources().unwrap();
    assert_eq!(factories(), 0);
    // Row storage is separately priced with its actual concrete layout. Its
    // factory need not run to price either it or the subsequent index owner.
    let row_bytes = layout_bytes::<Entry<&str, usize>>(stats.count()).unwrap();
    let mut rows = ExactBuffer::new(stats.count()).unwrap();
    for (position, key) in keys.into_iter().enumerate() {
        rows.push(Entry {
            key,
            position,
            value: position,
        })
        .unwrap();
    }
    let rows = rows.finish().unwrap();
    assert_eq!(row_bytes, std::mem::size_of_val(rows.as_ref()));
    assert_eq!(factories(), 1);
    let quote = Quote::checked(&rows, prequote.quote_work).unwrap();
    assert_eq!(quote.resources(), prequote);
    let index = quote
        .construct(prequote.workspace_bytes, prequote.construction_work)
        .unwrap();
    assert_eq!(factories(), 2);
    assert!(index.usage().work <= prequote.construction_work);
}

fn assert_long_queries_fit_stored_ceiling<K: Key>(entries: &[Entry<K, ()>], queries: &[K]) {
    let quote = Quote::checked(entries, u128::MAX).unwrap();
    let resources = quote.resources();
    let index = quote
        .construct(resources.workspace_bytes, resources.construction_work)
        .unwrap();
    let stored_key_bound = entries
        .iter()
        .map(|entry| {
            entry
                .key
                .quote_work(&mut Usage::default(), u128::MAX)
                .unwrap()
        })
        .max()
        .unwrap();
    reset_factories();
    for query in queries {
        // Establish that the query really exceeds every stored key's work span,
        // not merely that its enclosing allocation happens to have more capacity.
        let query_bound = query.quote_work(&mut Usage::default(), u128::MAX).unwrap();
        assert!(query_bound > stored_key_bound);
        let initial = Usage {
            work: 17,
            comparisons: 11,
        };
        let mut usage = initial;
        let exact_ceiling = initial.work.checked_add(resources.lookup_work).unwrap();
        let found = index.matches(query, &mut usage, exact_ceiling).unwrap();
        assert!(found.is_empty());
        assert!(usage.work > initial.work);
        assert!(usage.work - initial.work <= resources.lookup_work);
        assert!(usage.comparisons - initial.comparisons <= resources.lookup_comparisons);
    }
    assert_eq!(factories(), 0);
}

#[test]
fn longer_than_stored_strings_fit_exact_lookup_ceiling() {
    let entries: [Entry<&str, ()>; 6] =
        ["", "a", "prefix", "prefixa", "prefixé", "éé"].map(|key| Entry {
            key,
            position: 0,
            value: (),
        });
    let ascii = format!("prefixa{}", "x".repeat(4096));
    let utf8 = format!("éé{}", "λ💡".repeat(1024));
    assert_long_queries_fit_stored_ceiling(&entries, &[ascii.as_str(), utf8.as_str()]);
}

#[test]
fn longer_than_stored_string_lists_fit_exact_lookup_ceiling() {
    let stored: [Vec<String>; 4] = [
        vec![],
        vec!["shared".into()],
        vec!["shared".into(), "prefix".into()],
        vec!["shared".into(), "prefix".into(), "é".into()],
    ];
    let entries: Vec<_> = stored
        .iter()
        .enumerate()
        .map(|(position, key)| Entry {
            key: Strings(key),
            position,
            value: (),
        })
        .collect();
    let mut longer_list = stored[3].clone();
    longer_list.extend((0..64).map(|_| "extra".into()));
    let longer_first_string = vec![format!("shared{}", "λ".repeat(4096)); 65];
    assert!(stored.iter().all(|key| key.len() < longer_list.len()));
    assert!(stored
        .iter()
        .all(|key| key.len() < longer_first_string.len()));
    assert_long_queries_fit_stored_ceiling(
        &entries,
        &[Strings(&longer_list), Strings(&longer_first_string)],
    );
}

#[test]
fn longer_than_stored_nested_tuple_option_queries_fit_exact_lookup_ceiling() {
    let stored: [Vec<String>; 3] = [
        vec![],
        vec!["base".into()],
        vec!["base".into(), "tail".into()],
    ];
    let keys = [
        (None, (Scalar(0_u64), None)),
        (Some(("", Strings(&stored[0]))), (Scalar(0), None)),
        (
            Some(("same", Strings(&stored[1]))),
            (Scalar(1), Some("tail")),
        ),
        (
            Some(("same", Strings(&stored[2]))),
            (Scalar(1), Some("tail")),
        ),
    ];
    let entries: Vec<_> = keys
        .into_iter()
        .enumerate()
        .map(|(position, key)| Entry {
            key,
            position,
            value: (),
        })
        .collect();
    let long_string = format!("same{}", "💡".repeat(2048));
    let mut long_list = stored[2].clone();
    long_list.extend((0..64).map(|_| "extra".into()));
    let queries = [
        (
            Some((long_string.as_str(), Strings(&long_list))),
            (Scalar(1), Some(long_string.as_str())),
        ),
        (
            Some(("same", Strings(&long_list))),
            (Scalar(1), Some(long_string.as_str())),
        ),
        // Reach the final nested option after an entirely equal tuple/list prefix.
        (
            Some(("same", Strings(&stored[2]))),
            (Scalar(1), Some(long_string.as_str())),
        ),
    ];
    assert_long_queries_fit_stored_ceiling(&entries, &queries);
}

#[test]
fn last_before_ceiling_short_and_overflow_refusals_preserve_atomic_meter() {
    let entries = [7, 1, 3, 3, 5].map(|position| Entry {
        key: "same",
        position,
        value: (),
    });
    let quote = Quote::checked(&entries, u128::MAX).unwrap();
    let resources = quote.resources();
    let index = quote
        .construct(resources.workspace_bytes, resources.construction_work)
        .unwrap();
    let found = index
        .matches(&"same", &mut Usage::default(), u128::MAX)
        .unwrap();
    let comparisons = (usize::BITS - found.len().leading_zeros()) as u128;
    let bound = 2 * comparisons; // One driver + one scalar comparison per probe.
    let initial = Usage {
        work: 37,
        comparisons: 13,
    };
    reset_factories();
    for position in [0, 1, 2, 3, 4, 8, usize::MAX] {
        let mut short = initial;
        assert!(matches!(
            found.last_before(position, &mut short, initial.work + bound - 1),
            Err(Error::WorkLimit)
        ));
        assert_eq!(short, initial);
        let mut exact = initial;
        let actual = found
            .last_before(position, &mut exact, initial.work + bound)
            .unwrap();
        let expected = entries
            .iter()
            .filter(|entry| entry.position < position)
            .max_by_key(|entry| entry.position);
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(std::ptr::eq(actual, expected)),
            (None, None) => (),
            _ => panic!("predecessor mismatch for {position}"),
        }
        assert!(exact.work - initial.work <= bound);
        assert!(exact.comparisons - initial.comparisons <= comparisons);
        if position == 0 {
            assert_eq!(exact.work - initial.work, bound); // Exercise the ceiling itself.
        }
    }
    let mut zero = Usage::default();
    assert!(matches!(
        found.last_before(4, &mut zero, 0),
        Err(Error::WorkLimit)
    ));
    assert_eq!(zero, Usage::default());
    for initial in [
        Usage {
            work: u128::MAX - bound + 1,
            comparisons: 0,
        },
        Usage {
            work: 0,
            comparisons: u128::MAX - comparisons + 1,
        },
    ] {
        let mut usage = initial;
        assert!(matches!(
            found.last_before(4, &mut usage, u128::MAX),
            Err(Error::Overflow)
        ));
        assert_eq!(usage, initial);
    }
    assert_eq!(factories(), 0);
}

#[test]
fn surrogate_prequote_cannot_admit_larger_actual_rows_before_factory() {
    let mut surrogate = KeyStats::<&str, ()>::new();
    for _ in 0..4 {
        surrogate.push("s", u128::MAX).unwrap();
    }
    let surrogate = surrogate.resources().unwrap();
    // Already-existing caller-owned rows are outside the index quote; these
    // assertions concern the real permutation factory, not aggregate admission.
    let more_rows: [Entry<&str, ()>; 8] = std::array::from_fn(|position| Entry {
        key: "s",
        position,
        value: (),
    });
    let long_key = "same-original-key".repeat(64);
    let larger_keys: [Entry<&str, ()>; 4] = std::array::from_fn(|position| Entry {
        key: long_key.as_str(),
        position,
        value: (),
    });
    reset_factories();
    let actual = Quote::checked(&more_rows, u128::MAX).unwrap();
    let resources = actual.resources();
    assert!(resources.workspace_bytes > surrogate.workspace_bytes);
    assert!(matches!(
        actual.construct(surrogate.workspace_bytes, resources.construction_work),
        Err(Error::Capacity)
    ));
    assert_eq!(factories(), 0);
    let actual = Quote::checked(&larger_keys, u128::MAX).unwrap();
    let resources = actual.resources();
    assert_eq!(resources.workspace_bytes, surrogate.workspace_bytes);
    assert!(resources.construction_work > surrogate.construction_work);
    assert!(resources.lookup_work > surrogate.lookup_work);
    assert!(matches!(
        actual.construct(resources.workspace_bytes, surrogate.construction_work),
        Err(Error::WorkLimit)
    ));
    assert_eq!(factories(), 0);
    let index = Quote::checked(&larger_keys, u128::MAX)
        .unwrap()
        .construct(resources.workspace_bytes, resources.construction_work)
        .unwrap();
    assert_eq!(factories(), 1);
    assert!(index.usage().work <= resources.construction_work);
}

#[test]
fn initialized_prefix_drop_panic_cleans_remaining_elements_without_second_panic() {
    struct PanicOnce {
        ordinal: usize,
        seen: Rc<[Cell<usize>; 4]>,
        fired: Rc<Cell<bool>>,
    }
    impl Drop for PanicOnce {
        fn drop(&mut self) {
            self.seen[self.ordinal].set(self.seen[self.ordinal].get() + 1);
            let first_failure = self.ordinal == 1 && !self.fired.replace(true);
            assert!(!first_failure, "single element destructor failure");
        }
    }
    let seen = Rc::new(std::array::from_fn(|_| Cell::new(0)));
    let fired = Rc::new(Cell::new(false));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut buffer = ExactBuffer::new(4).unwrap();
        for ordinal in 0..3 {
            buffer
                .push(PanicOnce {
                    ordinal,
                    seen: seen.clone(),
                    fired: fired.clone(),
                })
                .unwrap();
        }
        // Normal drop starts outside any unwind. Exactly one initialized value
        // can panic; the fourth slot was never initialized and must not be dropped.
        drop(buffer);
    }));
    assert!(result.is_err());
    assert!(fired.get());
    assert_eq!(seen.each_ref().map(Cell::get), [1, 1, 1, 0]);
}
