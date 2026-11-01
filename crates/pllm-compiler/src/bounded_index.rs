//! Exact storage, metered heapsort and borrowed-key lookup shared by compiler owners.
//!
//! These are logical layout/work contracts, not Budget authority. An admitting
//! caller must hold its original reservation before invoking allocation factories.
//! Input owners, nested element backing, initializer/comparator state, allocator
//! overhead and dependency/native call stacks remain the enclosing owner's job.

use pllm_core::exact_buffer as backing;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Error {
    Overflow,
    Capacity,
    WorkLimit,
    Incomplete,
    Position,
}

pub(crate) fn checked_add(a: u128, b: u128) -> Result<u128, Error> {
    a.checked_add(b).ok_or(Error::Overflow)
}

pub(crate) fn checked_mul(a: u128, b: u128) -> Result<u128, Error> {
    a.checked_mul(b).ok_or(Error::Overflow)
}

pub(crate) fn layout_bytes<T>(count: usize) -> Result<usize, Error> {
    backing::layout_bytes::<T>(count).map_err(buffer_error)
}

fn buffer_error(error: backing::Error) -> Error {
    match error {
        backing::Error::Overflow => Error::Overflow,
        backing::Error::Allocation => Error::Capacity,
        backing::Error::Incomplete => Error::Incomplete,
    }
}

/// Explicit driver visits, comparisons, swaps and charged key spans. Not CPU
/// instructions. Sorting/search drivers increment `comparisons`; direct caller
/// comparisons need not do so (the runtime grouping contract counts heap calls).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Usage {
    pub(crate) work: u128,
    pub(crate) comparisons: u128,
}

impl Usage {
    // The driver first proves its checked total bound fits the counter. Keep this
    // identical to the original grouping meter, including its unit convention.
    pub(crate) fn step(&mut self) {
        self.work += 1;
    }

    pub(crate) fn charge(&mut self, work: u128, limit: u128) -> Result<(), Error> {
        let next = checked_add(self.work, work)?;
        if next > limit {
            return Err(Error::WorkLimit);
        }
        self.work = next;
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct SortState {
    root: usize,
    child: usize,
    end: usize,
}

/// Original runtime grouping heapsort, extracted without changing its driver,
/// comparison order or work units. No recursion, allocation or dependency sort.
pub(crate) fn heapsort<T>(
    items: &mut [T],
    usage: &mut Usage,
    mut compare: impl FnMut(&T, &T, &mut Usage) -> Ordering,
) {
    fn sift<T>(
        items: &mut [T],
        state: &mut SortState,
        usage: &mut Usage,
        compare: &mut impl FnMut(&T, &T, &mut Usage) -> Ordering,
    ) {
        while state.root < state.end / 2 {
            state.child = 2 * state.root + 1;
            if state.child + 1 < state.end {
                usage.comparisons += 1;
                usage.step();
                if compare(&items[state.child], &items[state.child + 1], usage).is_lt() {
                    state.child += 1;
                }
            }
            usage.comparisons += 1;
            usage.step();
            if !compare(&items[state.root], &items[state.child], usage).is_lt() {
                break;
            }
            items.swap(state.root, state.child);
            usage.step();
            state.root = state.child;
        }
    }
    let mut state = SortState {
        root: items.len() / 2,
        child: 0,
        end: items.len(),
    };
    while state.root > 0 {
        let root = state.root - 1;
        state.root = root;
        sift(items, &mut state, usage, &mut compare);
        state.root = root;
    }
    while state.end > 1 {
        state.end -= 1;
        items.swap(0, state.end);
        usage.step();
        state.root = 0;
        sift(items, &mut state, usage, &mut compare);
    }
}

pub(crate) fn sort_comparison_bound(n: usize) -> Result<u128, Error> {
    if n < 2 {
        return Ok(0);
    }
    let sifts = checked_add((n / 2) as u128, (n - 1) as u128)?;
    checked_mul(checked_mul(2, sifts)?, search_comparison_bound(n))
}

pub(crate) fn search_comparison_bound(n: usize) -> u128 {
    (usize::BITS - n.leading_zeros()) as u128
}

/// First item not less than the query. Caller owns comparator effects/work;
/// runtime uses this directly to preserve its existing ID-probe instrumentation.
pub(crate) fn lower_bound_by<T>(items: &[T], mut compare: impl FnMut(&T) -> Ordering) -> usize {
    let (mut lo, mut hi) = (0, items.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if compare(&items[mid]).is_lt() {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
thread_local! {
    static FACTORIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Compiler-local error/counter adapter over the shared exact prefix owner.
pub(crate) struct ExactBuffer<T> {
    backing: backing::ExactBuffer<T>,
}

impl<T> ExactBuffer<T> {
    pub(crate) fn new(count: usize) -> Result<Self, Error> {
        layout_bytes::<T>(count)?;
        #[cfg(test)]
        FACTORIES.with(|count| count.set(count.get() + 1));
        Ok(Self {
            backing: backing::ExactBuffer::new(count).map_err(buffer_error)?,
        })
    }

    /// Preserve the compiler's existing consuming refusal: a rejected incoming
    /// value is dropped here. The shared core API instead returns that value.
    pub(crate) fn push(&mut self, value: T) -> Result<(), Error> {
        self.backing.push(value).map_err(|_value| Error::Capacity)
    }

    pub(crate) fn finish(self) -> Result<Box<[T]>, Error> {
        self.backing.finish().map_err(buffer_error)
    }
}

/// Original grouping factory for Copy rows, including its exact allocation and
/// initializer order, using the same fallible backing constructor as non-Copy
/// metadata. Compiler Usage/accounting remains the enclosing caller's job.
pub(crate) fn exact_box<T: Copy>(
    count: usize,
    mut make: impl FnMut(usize) -> T,
) -> Result<Box<[T]>, Error> {
    let mut allocation = ExactBuffer::new(count)?;
    for index in 0..count {
        allocation.push(make(index))?;
    }
    allocation.finish()
}

// This complete keyed facade is exercised by focused tests; production runtime
// consumes the exact factory, sort/meter and lower-bound primitives above. The
// inventory owner will consume this facade in the next guarded-constructor step.
#[allow(
    dead_code,
    reason = "crate-private inventory index facade awaits its enclosing owner"
)]
pub(crate) mod keyed {
    use super::*;

    /// Key comparisons must charge at most either operand's quoted comparison
    /// bound. Built-in keys below inspect only borrowed strings or scalar fields.
    /// This compiler-local trait grants neither reservation nor source authority.
    pub(crate) trait Key: Copy {
        /// Concrete key-owned cursor state; native Ord/formatting frames are not
        /// included. Composite keys call one child at a time, never clone keys.
        fn cursor_bytes() -> usize;
        fn quote_work(&self, usage: &mut Usage, limit: u128) -> Result<u128, Error>;
        fn compare(&self, other: &Self, usage: &mut Usage) -> Ordering;
    }

    impl Key for &str {
        fn cursor_bytes() -> usize {
            0
        }
        fn quote_work(&self, usage: &mut Usage, limit: u128) -> Result<u128, Error> {
            usage.charge(1, limit)?;
            checked_add(self.len() as u128, 1)
        }
        fn compare(&self, other: &Self, usage: &mut Usage) -> Ordering {
            usage.work += self.len().min(other.len()) as u128 + 1;
            self.cmp(other)
        }
    }

    /// One scalar Ord invocation, suitable for existing enum/direction/layer
    /// fields. No discriminant remapping or new protocol numbering is introduced.
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
    pub(crate) struct Scalar<T>(pub(crate) T);

    /// Explicit owner audit for enum-like scalar fields. Do not admit borrowed
    /// collections/strings through Scalar and hide their span behind one unit.
    /// Client enums implement this marker without changing their original Ord.
    pub(crate) trait ScalarValue: Copy + Ord {}

    macro_rules! scalar_values {
        ($($ty:ty),* $(,)?) => { $(impl ScalarValue for $ty {})* };
    }
    scalar_values!(bool, char, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

    impl<T: ScalarValue> Key for Scalar<T> {
        fn cursor_bytes() -> usize {
            0
        }
        fn quote_work(&self, usage: &mut Usage, limit: u128) -> Result<u128, Error> {
            usage.charge(1, limit)?;
            Ok(1)
        }
        fn compare(&self, other: &Self, usage: &mut Usage) -> Ordering {
            usage.step();
            self.0.cmp(&other.0)
        }
    }

    impl<K: Key> Key for Option<K> {
        fn cursor_bytes() -> usize {
            K::cursor_bytes()
        }
        fn quote_work(&self, usage: &mut Usage, limit: u128) -> Result<u128, Error> {
            usage.charge(1, limit)?;
            checked_add(
                1,
                match self {
                    Some(key) => key.quote_work(usage, limit)?,
                    None => 0,
                },
            )
        }
        fn compare(&self, other: &Self, usage: &mut Usage) -> Ordering {
            usage.step();
            match (self, other) {
                (Some(a), Some(b)) => a.compare(b, usage),
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
            }
        }
    }

    impl<A: Key, B: Key> Key for (A, B) {
        fn cursor_bytes() -> usize {
            A::cursor_bytes().max(B::cursor_bytes())
        }
        fn quote_work(&self, usage: &mut Usage, limit: u128) -> Result<u128, Error> {
            usage.charge(1, limit)?;
            checked_add(
                self.0.quote_work(usage, limit)?,
                self.1.quote_work(usage, limit)?,
            )
        }
        fn compare(&self, other: &Self, usage: &mut Usage) -> Ordering {
            self.0
                .compare(&other.0, usage)
                .then_with(|| self.1.compare(&other.1, usage))
        }
    }

    impl<A: Key, B: Key, C: Key> Key for (A, B, C) {
        fn cursor_bytes() -> usize {
            A::cursor_bytes()
                .max(B::cursor_bytes())
                .max(C::cursor_bytes())
        }
        fn quote_work(&self, usage: &mut Usage, limit: u128) -> Result<u128, Error> {
            usage.charge(1, limit)?;
            checked_add(
                checked_add(
                    self.0.quote_work(usage, limit)?,
                    self.1.quote_work(usage, limit)?,
                )?,
                self.2.quote_work(usage, limit)?,
            )
        }
        fn compare(&self, other: &Self, usage: &mut Usage) -> Ordering {
            self.0
                .compare(&other.0, usage)
                .then_with(|| self.1.compare(&other.1, usage))
                .then_with(|| self.2.compare(&other.2, usage))
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Strings<'a>(pub(crate) &'a [String]);

    struct StringsQuoteState<'a> {
        values: std::slice::Iter<'a, String>,
        bound: u128,
    }

    struct StringsCompareState<'a, 'b> {
        values: std::iter::Zip<std::slice::Iter<'a, String>, std::slice::Iter<'b, String>>,
    }

    impl Key for Strings<'_> {
        fn cursor_bytes() -> usize {
            std::mem::size_of::<StringsQuoteState<'_>>()
                .max(std::mem::size_of::<StringsCompareState<'_, '_>>())
        }
        fn quote_work(&self, usage: &mut Usage, limit: u128) -> Result<u128, Error> {
            usage.charge(1, limit)?;
            let mut state = StringsQuoteState {
                values: self.0.iter(),
                bound: 1, // Length comparison after the common prefix.
            };
            for value in state.values.by_ref() {
                usage.charge(1, limit)?;
                state.bound = checked_add(state.bound, checked_add(value.len() as u128, 1)?)?;
            }
            Ok(state.bound)
        }
        fn compare(&self, other: &Self, usage: &mut Usage) -> Ordering {
            let state = StringsCompareState {
                values: self.0.iter().zip(other.0),
            };
            for (a, b) in state.values {
                let order = a.as_str().compare(&b.as_str(), usage);
                if !order.is_eq() {
                    return order;
                }
            }
            usage.step();
            self.0.len().cmp(&other.0.len())
        }
    }

    /// The input slice stays immutable and caller-owned. `position` is validation
    /// chronology, not a protocol ID. Equal positions tie by original slice index.
    #[derive(Debug)]
    pub(crate) struct Entry<K, V> {
        pub(crate) key: K,
        pub(crate) position: usize,
        pub(crate) value: V,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) struct Resources {
        pub(crate) heap_bytes: usize,
        pub(crate) workspace_bytes: usize,
        pub(crate) quote_work: u128,
        pub(crate) construction_work: u128,
        pub(crate) construction_comparisons: u128,
        pub(crate) lookup_work: u128,
        pub(crate) lookup_comparisons: u128,
    }

    /// Allocation-free quote over the very same immutable input slice. Existing
    /// entry/key/value backing is excluded, not replaced by a guessed baseline.
    #[derive(Debug)]
    pub(crate) struct Quote<'a, K, V> {
        entries: &'a [Entry<K, V>],
        resources: Resources,
    }

    #[derive(Debug)]
    pub(crate) struct Index<'a, K, V> {
        entries: &'a [Entry<K, V>],
        order: Box<[usize]>,
        resources: Resources,
        usage: Usage,
    }

    /// Allocation-free owner prepass before derived Entry storage even exists.
    /// Enclosing owner projects original borrowed keys under its own bounded scan
    /// and keeps their immutable source borrow. This accumulator has no factory
    /// or identity/reservation authority; Quote rechecks the actual entry slice.
    pub(crate) struct KeyStats<K, V> {
        count: usize,
        key_work: u128,
        usage: Usage,
        types: std::marker::PhantomData<fn() -> (K, V)>,
    }

    impl<K: Key, V> KeyStats<K, V> {
        pub(crate) fn new() -> Self {
            Self {
                count: 0,
                key_work: 0,
                usage: Usage::default(),
                types: std::marker::PhantomData,
            }
        }

        pub(crate) fn push(&mut self, key: K, max_quote_work: u128) -> Result<(), Error> {
            self.usage.charge(1, max_quote_work)?;
            let key_work = key.quote_work(&mut self.usage, max_quote_work)?;
            let count = self.count.checked_add(1).ok_or(Error::Overflow)?;
            self.count = count;
            self.key_work = self.key_work.max(key_work);
            Ok(())
        }

        pub(crate) fn count(&self) -> usize {
            self.count
        }

        pub(crate) fn resources(&self) -> Result<Resources, Error> {
            let heap_bytes = layout_bytes::<usize>(self.count)?;
            let comparisons = sort_comparison_bound(self.count)?;
            // Sort call + key + position + original index + at most one sift
            // swap per comparison, plus exactly n-1 extraction swaps. The key
            // quote supplies key units; the four others are driver/tie/swap units.
            let construction_work = checked_add(
                self.count as u128, // Exact permutation initialization.
                checked_add(
                    checked_mul(comparisons, checked_add(self.key_work, 4)?)?,
                    self.count.saturating_sub(1) as u128,
                )?,
            )?;
            let lookup_comparisons = checked_mul(2, search_comparison_bound(self.count))?;
            let lookup_work = checked_mul(lookup_comparisons, checked_add(self.key_work, 1)?)?;
            // Actual concrete owner/result/cursor slots. No entry clones or
            // per-key/per-run storage. Nested caller-owned backing is excluded.
            let mut workspace_bytes = heap_bytes;
            for bytes in [
                std::mem::size_of::<Self>(),
                std::mem::size_of::<Quote<'_, K, V>>(),
                std::mem::size_of::<Index<'_, K, V>>(),
                std::mem::size_of::<ExactBuffer<usize>>(),
                std::mem::size_of::<Result<ExactBuffer<usize>, Error>>(),
                std::mem::size_of::<Result<(), Error>>(),
                std::mem::size_of::<Result<u128, Error>>(),
                std::mem::size_of::<K>(),
                std::mem::size_of::<Result<Resources, Error>>(),
                std::mem::size_of::<usize>(),
                std::mem::size_of::<std::ops::Range<usize>>(),
                std::mem::size_of::<Result<Box<[usize]>, Error>>(),
                std::mem::size_of::<Result<Quote<'_, K, V>, Error>>(),
                std::mem::size_of::<Result<Index<'_, K, V>, Error>>(),
                std::mem::size_of::<SortState>(),
                std::mem::size_of::<Usage>(),
                K::cursor_bytes(),
                std::mem::size_of::<std::slice::Iter<'_, Entry<K, V>>>(),
                std::mem::size_of::<Matches<'_, '_, K, V>>(),
                std::mem::size_of::<MatchIter<'_, '_, K, V>>(),
                std::mem::size_of::<Result<Matches<'_, '_, K, V>, Error>>(),
                std::mem::size_of::<Result<Option<&Entry<K, V>>, Error>>(),
                std::mem::size_of::<std::alloc::Layout>(),
                std::mem::size_of::<Result<std::alloc::Layout, std::alloc::LayoutError>>(),
            ] {
                workspace_bytes = workspace_bytes.checked_add(bytes).ok_or(Error::Overflow)?;
            }
            Ok(Resources {
                heap_bytes,
                workspace_bytes,
                quote_work: self.usage.work,
                construction_work,
                construction_comparisons: comparisons,
                lookup_work,
                lookup_comparisons,
            })
        }
    }

    impl<'a, K: Key, V> Quote<'a, K, V> {
        /// Meter each key/collection visit before traversing it. `max_quote_work`
        /// bounds this prepass; construction has its own separately checked bound.
        pub(crate) fn checked(
            entries: &'a [Entry<K, V>],
            max_quote_work: u128,
        ) -> Result<Self, Error> {
            layout_bytes::<usize>(entries.len())?;
            let mut stats = KeyStats::<K, V>::new();
            for entry in entries {
                stats.push(entry.key, max_quote_work)?;
            }
            Ok(Self {
                entries,
                resources: stats.resources()?,
            })
        }

        pub(crate) fn resources(&self) -> Resources {
            self.resources
        }

        #[cfg(test)]
        pub(super) fn original_entries(&self) -> &'a [Entry<K, V>] {
            self.entries
        }

        /// Numeric ceilings are checks, NOT a reservation. Enclosing owner holds
        /// the real original guard over inputs, workspace and every return path.
        pub(crate) fn construct(
            self,
            workspace_capacity: usize,
            max_work: u128,
        ) -> Result<Index<'a, K, V>, Error> {
            if workspace_capacity < self.resources.workspace_bytes {
                return Err(Error::Capacity);
            }
            if max_work < self.resources.construction_work {
                return Err(Error::WorkLimit);
            }
            let mut usage = Usage::default();
            let mut allocation = ExactBuffer::new(self.entries.len())?;
            for index in 0..self.entries.len() {
                usage.step();
                allocation.push(index)?;
            }
            let mut order = allocation.finish()?;
            heapsort(&mut order, &mut usage, |a, b, usage| {
                self.entries[*a]
                    .key
                    .compare(&self.entries[*b].key, usage)
                    .then_with(|| {
                        usage.step();
                        self.entries[*a].position.cmp(&self.entries[*b].position)
                    })
                    .then_with(|| {
                        usage.step();
                        a.cmp(b)
                    })
            });
            Ok(Index {
                entries: self.entries,
                order,
                resources: self.resources,
                usage,
            })
        }
    }

    fn require_work(
        usage: &Usage,
        work: u128,
        comparisons: u128,
        limit: u128,
    ) -> Result<(), Error> {
        let work = checked_add(usage.work, work)?;
        checked_add(usage.comparisons, comparisons)?;
        if work > limit {
            return Err(Error::WorkLimit);
        }
        Ok(())
    }

    impl<'a, K: Key, V> Index<'a, K, V> {
        #[cfg(test)]
        pub(super) fn ordered_ordinals(&self) -> &[usize] {
            &self.order
        }

        pub(crate) fn resources(&self) -> Resources {
            self.resources
        }

        pub(crate) fn usage(&self) -> Usage {
            self.usage
        }

        /// Two binary searches, even for long duplicate runs. Range iteration
        /// visits only requested matches; first/last/precedence never rescan.
        pub(crate) fn matches(
            &self,
            key: &K,
            usage: &mut Usage,
            max_work: u128,
        ) -> Result<Matches<'_, 'a, K, V>, Error> {
            require_work(
                usage,
                self.resources.lookup_work,
                self.resources.lookup_comparisons,
                max_work,
            )?;
            let mut compare = |index: &usize| {
                usage.comparisons += 1;
                usage.step();
                self.entries[*index].key.compare(key, usage)
            };
            let start = lower_bound_by(&self.order, &mut compare);
            // Upper bound expressed through the same lower-bound driver: equal
            // items are less than the exclusive end, greater items are not.
            let end = start
                + lower_bound_by(&self.order[start..], |index| match compare(index) {
                    Ordering::Less | Ordering::Equal => Ordering::Less,
                    Ordering::Greater => Ordering::Greater,
                });
            Ok(Matches {
                index: self,
                range: start..end,
            })
        }

        pub(crate) fn matches_for_input(
            &self,
            ordinal: usize,
            usage: &mut Usage,
            max_work: u128,
        ) -> Result<Matches<'_, 'a, K, V>, Error> {
            let entry = self.entries.get(ordinal).ok_or(Error::Position)?;
            self.matches(&entry.key, usage, max_work)
        }
    }

    pub(crate) struct Matches<'i, 'a, K, V> {
        index: &'i Index<'a, K, V>,
        range: std::ops::Range<usize>,
    }

    impl<'a, K, V> Matches<'_, 'a, K, V> {
        pub(crate) fn len(&self) -> usize {
            self.range.len()
        }

        pub(crate) fn is_empty(&self) -> bool {
            self.range.is_empty()
        }

        pub(crate) fn first(&self) -> Option<&'a Entry<K, V>> {
            self.index.order[self.range.clone()]
                .first()
                .map(|&index| &self.index.entries[index])
        }

        pub(crate) fn last(&self) -> Option<&'a Entry<K, V>> {
            self.index.order[self.range.clone()]
                .last()
                .map(|&index| &self.index.entries[index])
        }

        pub(crate) fn first_before(&self, position: usize) -> Option<&'a Entry<K, V>> {
            self.first().filter(|entry| entry.position < position)
        }

        /// Validation chronology ties by input ordinal, not sort instability.
        pub(crate) fn precedes(&self, position: usize, ordinal: usize) -> bool {
            self.index.order[self.range.clone()]
                .first()
                .is_some_and(|&first| {
                    (self.index.entries[first].position, first) < (position, ordinal)
                })
        }

        pub(crate) fn last_before(
            &self,
            position: usize,
            usage: &mut Usage,
            max_work: u128,
        ) -> Result<Option<&'a Entry<K, V>>, Error> {
            let comparisons = search_comparison_bound(self.len());
            require_work(usage, checked_mul(2, comparisons)?, comparisons, max_work)?;
            let items = &self.index.order[self.range.clone()];
            let end = lower_bound_by(items, |&index| {
                usage.comparisons += 1;
                usage.step(); // Driver comparison.
                usage.step(); // Scalar position comparison.
                self.index.entries[index].position.cmp(&position)
            });
            Ok(end
                .checked_sub(1)
                .map(|last| &self.index.entries[items[last]]))
        }

        pub(crate) fn iter(&self) -> MatchIter<'_, 'a, K, V> {
            MatchIter {
                entries: self.index.entries,
                order: self.index.order[self.range.clone()].iter(),
            }
        }
    }

    pub(crate) struct MatchIter<'i, 'a, K, V> {
        entries: &'a [Entry<K, V>],
        order: std::slice::Iter<'i, usize>,
    }

    impl<'a, K, V> Iterator for MatchIter<'_, 'a, K, V> {
        type Item = &'a Entry<K, V>;
        fn next(&mut self) -> Option<Self::Item> {
            self.order.next().map(|&index| &self.entries[index])
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.order.size_hint()
        }
    }
    impl<K, V> ExactSizeIterator for MatchIter<'_, '_, K, V> {}
}

#[cfg(test)]
#[path = "bounded_index_tests.rs"]
mod tests;
