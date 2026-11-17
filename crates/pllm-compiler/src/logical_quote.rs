//! Compiler-private inert K+V+P rows, including local logical cell retirement.
//! Not an allocator, drop-glue, residence, CPU, admission or funding contract.

use std::mem::{align_of, size_of};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cell {
    pub(crate) layout: [usize; 2],
    pub(crate) loan: [usize; 2],
    pub(crate) tag: bool,
}
pub(crate) const EMPTY: Cell = Cell { layout: [0, 1], loan: [0, 1], tag: false };
pub(crate) const fn cell<T>(tag: bool) -> Cell {
    Cell { layout: [size_of::<T>(), align_of::<T>()],
        loan: [size_of::<&mut T>(), align_of::<&mut T>()], tag }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Event {
    pub(crate) id: &'static str,
    pub(crate) kind: &'static str,
    pub(crate) kvp: [u128; 3],
    pub(crate) cells: [Cell; 4],
    pub(crate) count: usize,
}
// Operand tuples retain padding. References to enums are not tagged enums.
pub(crate) const fn call<A, T>(id: &'static str, tag: bool) -> Event {
    Event { id, kind: "C", kvp: [1, 1, 2 * (size_of::<A>() as u128 + size_of::<T>() as u128)],
        cells: [cell::<A>(false), cell::<A>(false), cell::<T>(tag), cell::<T>(tag)], count: 4 }
}
pub(crate) const fn bind<T>(id: &'static str, tag: bool) -> Event {
    Event { id, kind: "B", kvp: [0, 1, 2 * size_of::<T>() as u128],
        cells: [cell::<T>(tag), cell::<T>(tag), EMPTY, EMPTY], count: 2 }
}
pub(crate) const fn branch<T>(id: &'static str, tag: bool) -> Event {
    Event { id, kind: "H", kvp: [0, 1, size_of::<T>() as u128 + 2 * size_of::<bool>() as u128],
        cells: [cell::<T>(tag), cell::<bool>(false), EMPTY, EMPTY], count: 2 }
}
pub(crate) const fn transfer<T>(id: &'static str, n: u128) -> Event {
    // All callers use fixed, small occurrence counts, not external multipliers.
    Event { id, kind: "transfer", kvp: [0, 0, n * size_of::<T>() as u128], cells: [EMPTY; 4], count: 0 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuoteError { UnsupportedLayout, Overflow }
macro_rules! checked {
    ($e:expr) => { match $e { Some(n) => n, None => return Err(QuoteError::Overflow) } };
}
pub(crate) use checked;

pub(crate) const fn event_work(e: Event) -> Result<u128, QuoteError> {
    let mut out = checked!(checked!(e.kvp[0].checked_add(e.kvp[1])).checked_add(e.kvp[2]));
    let mut i = 0;
    while i < e.count {
        let z = checked!(checked!(1u128.checked_add(e.cells[i].tag as u128))
            .checked_add(checked!(2u128.checked_mul(e.cells[i].loan[0] as u128))));
        out = checked!(out.checked_add(z));
        i += 1;
    }
    Ok(out)
}
pub(crate) const fn fold(events: &[Event]) -> Result<u128, QuoteError> {
    let mut out = 0u128;
    let mut i = 0;
    while i < events.len() {
        let w = match event_work(events[i]) { Ok(w) => w, Err(e) => return Err(e) };
        out = checked!(out.checked_add(w));
        i += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_ports_and_tagged_cell_retirement_are_charged_separately() {
        type Args = (u8, u64);
        type Output = Result<u64, u8>;
        let args = size_of::<Args>() as u128;
        let output = size_of::<Output>() as u128;
        let args_loan = size_of::<&mut Args>() as u128;
        let output_loan = size_of::<&mut Output>() as u128;
        assert!(args > (size_of::<u8>() + size_of::<u64>()) as u128);
        let invoke = call::<Args, Output>("invoke", true);
        assert_eq!(invoke.kvp, [1, 1, 2 * (args + output)]);
        assert_eq!(invoke.cells.map(|c| c.tag), [false, false, true, true]);
        let invoke_work = 2 + 2 * (args + output) + 2 * (1 + 2 * args_loan)
            + 2 * (2 + 2 * output_loan);
        let forward = transfer::<Output>("forward", 2);
        assert_eq!(event_work(invoke), Ok(invoke_work));
        assert_eq!(event_work(forward), Ok(2 * output));
        assert_eq!(fold(&[invoke, forward]), Ok(invoke_work + 2 * output));
    }

    #[test]
    fn rejects_overflow_within_row_retirement_and_across_rows() {
        let max = Event { id: "max", kind: "test", kvp: [0, 0, u128::MAX],
            cells: [EMPTY; 4], count: 0 };
        assert_eq!(event_work(max), Ok(u128::MAX));
        assert_eq!(event_work(Event { kvp: [1, 0, u128::MAX], ..max }), Err(QuoteError::Overflow));
        assert_eq!(event_work(Event { count: 1, ..max }), Err(QuoteError::Overflow));
        assert_eq!(fold(&[]), Ok(0));
        assert_eq!(fold(&[max, transfer::<u8>("one", 1)]), Err(QuoteError::Overflow));
    }
}
