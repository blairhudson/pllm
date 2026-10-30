use super::*;

fn builder(budget: &Budget) -> Builder<'_> {
    Builder::new([7; 32], b"stored-lookup-census", &[10], budget).unwrap()
}

fn lookup_plan(budget: &Budget) -> Plan<'_> {
    let mut b = builder(budget);
    let table = b.table(10, &[1; 513]).unwrap();
    let chunks = b.tables[table].resources(1).unwrap().chunks_per_lane;
    assert!(chunks > 1);
    let a = b.lookup(0, table).unwrap();
    let z = b.lookup(a, table).unwrap();
    assert_eq!(b.lookup_chunks, 2 * chunks);
    let plan = b.finish(&[z]).unwrap();
    assert_eq!(plan.stored_lookup_chunks(), 2 * chunks);
    plan
}

#[test]
fn census_counts_stored_occurrences_not_distinct_tables_or_virtual_lanes() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut b = builder(&budget);
    assert_eq!(b.table(10, &[]), Err(Error::Domain));
    b.table(10, &[1; 513]).unwrap(); // Retained but unused is not a visit.
    assert_eq!(b.finish(&[0]).unwrap().stored_lookup_chunks(), 0);
    let scalar = lookup_plan(&budget);
    let chunks = scalar.stored_lookup_chunks();
    drop(scalar);
    let expanded = lookup_plan(&budget).repeat(3).unwrap();
    let tensor = lookup_plan(&budget).repeat_tensor(3).unwrap();
    assert_eq!(expanded.stored_lookup_chunks(), 3 * chunks);
    assert_eq!(tensor.stored_lookup_chunks(), chunks);
    assert_eq!(expanded.resources().instructions, tensor.resources().instructions);
    drop(expanded);
    drop(tensor);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn rejected_lookup_leaves_census_and_stored_code_unchanged() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut b = builder(&budget);
    let table = b.table(10, &[1; 513]).unwrap();
    b.lookup(0, table).unwrap();
    let chunks = b.lookup_chunks;
    let code = b.code.len();
    assert_eq!(b.lookup(usize::MAX, table), Err(Error::Domain));
    assert_eq!(b.lookup(0, usize::MAX), Err(Error::Domain));
    assert_eq!(b.lookup_chunks, chunks);
    assert_eq!(b.code.len(), code);
    // A private overflow fixture checks prospective census before any push.
    b.lookup_chunks = usize::MAX;
    let work = b.resources.cumulative_work;
    assert_eq!(b.lookup(0, table), Err(Error::Resource));
    assert_eq!(b.lookup_chunks, usize::MAX);
    assert_eq!(b.resources.cumulative_work, work);
    assert_eq!(b.code.len(), code);
    b.lookup_chunks = chunks;
    b.resources.cumulative_work = MAX_WORK;
    assert_eq!(b.lookup(0, table), Err(Error::Resource));
    assert_eq!(b.lookup_chunks, chunks);
    assert_eq!(b.code.len(), code);
    // Capacity refusal also cannot commit a census increment.
    b.resources.cumulative_work = work;
    b.code.resize(MAX_INSTRUCTIONS, Instruction::Constant(0));
    assert_eq!(b.lookup(0, table), Err(Error::Resource));
    assert_eq!(b.lookup_chunks, chunks);
    drop(b);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn repetition_overflow_or_refusal_never_exports_partial_census() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut plan = lookup_plan(&budget);
    plan.lookup_chunks = usize::MAX;
    // repeat consumes on error: no Plan or modified census escapes. Its owners
    // must retire, including when prospective census multiplication overflows.
    assert!(matches!(plan.repeat(2), Err(Error::Resource)));
    assert_eq!(budget.used_bytes(), 0);
    let plan = lookup_plan(&budget);
    assert!(matches!(plan.repeat(0), Err(Error::Resource)));
    assert_eq!(budget.used_bytes(), 0);
    let plan = lookup_plan(&budget);
    assert!(matches!(plan.repeat_tensor(0), Err(Error::Resource)));
    assert_eq!(budget.used_bytes(), 0);
}
