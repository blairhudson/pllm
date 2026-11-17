//! Private structural facts over the actual immutable DAG. No input opening,
//! caller-asserted interval, transferable certificate, or activation authority.
//! Facts remain conditional on the root's existing public-domain obligation;
//! they neither inspect membership nor totalize unchecked private inputs.
use super::*;
use super::attention_work::SoftmaxSink;
use super::super::arithmetic::IndexedTerm;
use crate::exact_buffer::linked_rows;
type Records = linked_rows::Chain<IndexedAffineRow, Value, LaneProof>;
type RecordOwner = Box<dyn linked_rows::Link<IndexedAffineRow, Value, LaneProof>>;
pub(super) type RecordLink = dyn linked_rows::Link<IndexedAffineRow, Value, LaneProof>;

const EXTENSION: &[u8] = b"puma.indexed-predicate-provenance.v1";
pub(super) const CAPACITY: usize = 384;

// Inline: absence owns no table/rule/proof headers and needs no allocation.
// Its physical size is still the enum's size, not a smaller Program layout.
// Legacy source-dispatch census for Rows identity: capacity/owners/owners_mut/
// backing_bytes (4), Dag extended/table/rule/proofs (4), finish validation/cache
// (2), hash_header/hash_rule (2), resources validation (1): thirteen EXISTING
// dispatch locations, thirteen after extension. Their Absent/Indexed bodies
// are preserved; replacing !absent predicates removes helper calls, adds none.
// absent(), indexed table/owner lookups, hash_rule's Indexed-only let/else and
// actual-Absent RMS/proof-free guards retain their contracts. Old constructors
// still initialize only Extension::Absent. No new Program field/None write,
// old-path identity validator/hash update/cache walk, or spare 21-control claim.
// Rows conversion/geometry bodies are exclusive new arms with separate K rows.
#[derive(Clone)]
pub(super) enum Extension {
    Absent,
    Indexed(IndexedOwners),
    CheckedRowsIdentity(CheckedRowsIdentity),
}

pub(super) struct IndexedOwners {
    pub(super) capacity: usize,
    pub(super) records: Records,
    pub(super) rules: [Option<Rule>; MAX_NODES],
    pub(super) backing_bytes: usize,
    pub(super) check_visits: u64,
}

impl IndexedOwners {
    pub(super) fn empty() -> Self {
        #[cfg(test)]
        OWNER_INITIALIZATIONS.with(|n| n.set(n.get() + 1));
        Self { capacity: CAPACITY, records: Records::empty(),
            rules: [None; MAX_NODES],
            backing_bytes: 0, check_visits: 0 }
    }
    pub(super) fn table(&self, index: usize) -> &[IndexedAffineRow] {
        if index >= MAX_NODES { return &[]; }
        match self.records.find(Value(index as u8)) { Some(link) => link.rows(), None => &[] }
    }
    pub(super) fn proofs(&self, index: usize) -> &[LaneProof] {
        if index >= MAX_NODES { return &[]; }
        match self.records.find(Value(index as u8)) { Some(link) => link.proofs(), None => &[] }
    }
    #[cfg(test)]
    pub(super) fn table_mut(&mut self, index: usize) -> &mut [IndexedAffineRow] {
        self.records.find_mut(Value(index as u8)).unwrap().rows_mut()
    }
    #[cfg(test)]
    pub(super) fn proofs_mut(&mut self, index: usize) -> &mut [LaneProof] {
        self.records.find_mut(Value(index as u8)).unwrap().proofs_slot()
    }
    #[cfg(test)]
    pub(super) fn insert_record_for_test(&mut self, node: Value, rows: &[IndexedAffineRow]) {
        let mut link = linked_rows::allocate(rows.len(), IndexedAffineRow::default(), node).unwrap();
        for row in rows { link.push(*row).unwrap(); }
        self.backing_bytes += link.allocation_bytes();
        self.records.prepend(link);
    }
}

impl Clone for IndexedOwners {
    fn clone(&self) -> Self {
        #[cfg(test)]
        OWNER_CLONES.with(|n| n.set(n.get() + 1));
        Self { capacity: self.capacity, records: self.records.clone(), rules: self.rules,
            backing_bytes: self.backing_bytes, check_visits: self.check_visits }
    }
}

#[cfg(test)]
thread_local! {
    static OWNER_INITIALIZATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static OWNER_CLONES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static OWNER_DROPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CHECKER_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static DESCRIPTOR_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
impl Drop for IndexedOwners {
    fn drop(&mut self) { OWNER_DROPS.with(|n| n.set(n.get() + 1)); }
}
#[cfg(test)]
pub(super) fn access_counts() -> [usize; 5] {
    [OWNER_INITIALIZATIONS.with(|n| n.get()), OWNER_CLONES.with(|n| n.get()),
        OWNER_DROPS.with(|n| n.get()), CHECKER_ENTRIES.with(|n| n.get()), DESCRIPTOR_READS.with(|n| n.get())]
}

impl Extension {
    pub(super) fn absent(&self) -> bool { matches!(self, Self::Absent) }
    pub(super) fn capacity(&self, lanes: usize) -> usize {
        match self { Self::Absent | Self::CheckedRowsIdentity(_) => lanes, Self::Indexed(o) => o.capacity }
    }
    pub(super) fn owners(&self) -> Result<&IndexedOwners, Error> {
        match self { Self::Absent | Self::CheckedRowsIdentity(_) => Err(Error::Phase), Self::Indexed(o) => Ok(o) }
    }
    pub(super) fn owners_mut(&mut self) -> Result<&mut IndexedOwners, Error> {
        match self { Self::Absent | Self::CheckedRowsIdentity(_) => Err(Error::Phase), Self::Indexed(o) => Ok(o) }
    }
    pub(super) fn backing_bytes(&self) -> usize {
        match self { Self::Absent | Self::CheckedRowsIdentity(_) => 0, Self::Indexed(o) => o.backing_bytes }
    }
    pub(super) fn table(&self, index: usize) -> Result<&[IndexedAffineRow], Error> {
        #[cfg(test)]
        DESCRIPTOR_READS.with(|n| n.set(n.get() + 1));
        if index >= MAX_NODES { return Err(Error::Binding); }
        Ok(self.owners()?.table(index))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rule {
    NestedDifference { packed: Value },
    MaskedOperand { original: Value, inside: Value },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Predicate {
    root: Value,
    lane: usize,
    offset: Value,
    sign: Value,
    b2a: Value,
    packed_lane: usize,
    threshold: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LaneProof {
    low: Predicate,
    high: Predicate,
}

/// Actual checker-walk accounting, distinct from native ordinals/PRSS. Each
/// helper records its source access/control block. Arithmetic validation uses
/// a conservative 64-visit block per row/term (including iterator transitions).
/// This is an inventory, never a source payer, admission permit or fee.
#[derive(Default)]
pub(super) struct Audit { pub(super) visits: u64 }

impl Audit {
    pub(super) fn add(&mut self, visits: usize) -> Result<(), Error> {
        self.visits = self.visits.checked_add(u64::try_from(visits).map_err(|_| Error::Bounds)?)
            .ok_or(Error::Bounds)?;
        Ok(())
    }
    pub(super) fn affine(&mut self, rows: &[IndexedAffineRow]) -> Result<(), Error> {
        self.add(64)?;
        for row in rows { self.add(64 * (1 + usize::from(row.len).min(2)))?; }
        Ok(())
    }
}

// At most 192 nodes, 1024 rows each. Per row: two direct five-node
// predicates, ordered affine checks, witness comparison, masked hull and hash
// controls fit 4096 source visits. Masked rules recheck one nested rule, never
// recurse through arbitrary proofs. Includes the independent full table walk.
pub(super) const MAX_VALIDATION_VISITS: u64 =
    MAX_NODES as u64 * (1024 * 4096 + 1024);

pub(super) fn checking_scratch_bytes() -> usize {
    // Named aggregate slots in validate -> masked -> nested -> predicate;
    // no lane-sized scratch, recursion, secret storage or interpreter.
    12 * std::mem::size_of::<Node>()
        + 8 * std::mem::size_of::<Predicate>()
        + 6 * std::mem::size_of::<LaneProof>()
        + 12 * std::mem::size_of::<Interval>()
        + 8 * std::mem::size_of::<[i128; 2]>()
        + std::mem::size_of::<Sha256>()
        + std::mem::size_of::<Audit>()
        + cache_build_scratch_bytes()
        + std::mem::size_of::<CachedDag<'static>>()
        + std::mem::size_of::<Result<CachedDag<'static>, Error>>()
        + lookup_control_bytes()
}

pub(super) const fn lookup_control_bytes() -> usize {
    // Occasional runtime lookup, NOT repeated inside a predicate/row checker.
    // Caller return/binding plus Chain::find current/next/identity controls.
    3 * std::mem::size_of::<Option<&'static RecordLink>>()
        + 2 * std::mem::size_of::<&'static RecordLink>()
        + 2 * std::mem::size_of::<&'static Records>()
        + 2 * std::mem::size_of::<&'static IndexedOwners>()
        + std::mem::size_of::<&'static Extension>()
        + 2 * std::mem::size_of::<usize>() + 2 * std::mem::size_of::<Value>()
        + std::mem::size_of::<&'static [IndexedAffineRow]>()
        + std::mem::size_of::<Result<&'static [IndexedAffineRow], Error>>()
        + std::mem::size_of::<Result<&'static IndexedOwners, Error>>()
}

/// Logical owner inventory, NOT Construction/FactoryParameters authority or an
/// allocator/RSS guarantee. Native and metadata phases are disjoint; do not add
/// them as if both must coexist. Enclosing caller/ABI/source payer remains out
/// of scope. Caller descriptor backing is explicitly retained during copying.
pub(super) struct IndexedInventory {
    pub(super) descriptor_bytes: usize,
    pub(super) proof_bytes: usize,
    pub(super) caller_descriptor_bytes: usize,
    pub(super) pending_owner_bytes: usize,
    pub(super) allocation_scratch_bytes: usize,
    pub(super) checker_scratch_bytes: usize,
    pub(super) finish_scratch_bytes: usize,
    pub(super) native_known_peak_bytes: usize,
    pub(super) metadata_known_peak_bytes: usize,
    pub(super) allocation_attempts: usize,
    pub(super) finish_validation_visits: u64,
    pub(super) hash_visits_bound: u64,
}

impl IndexedInventory {
    /// Each pair is (output rows, checked nested rule). No descriptor generation,
    /// compilation, private input, native factory or proof issuance occurs here.
    pub(super) fn quote(lanes: usize, maps: &[(usize, bool)]) -> Result<Self, Error> {
        validate_tile_lanes(lanes)?;
        if maps.is_empty() || maps.len() >= MAX_NODES { return Err(Error::Bounds); }
        let mut descriptor_bytes = 0usize;
        let mut proof_bytes = 0usize;
        let mut allocation_attempts = 0usize;
        let mut caller_descriptor_bytes = 0usize;
        let mut staging = 0usize;
        let mut rows_total = 0u64;
        let mut proofs_total = 0u64;
        for &(rows, proof) in maps {
            if rows == 0 || rows > CAPACITY { return Err(Error::Bounds); }
            descriptor_bytes = descriptor_bytes.checked_add(backing_bytes(rows, 0)?).ok_or(Error::Bounds)?;
            caller_descriptor_bytes = caller_descriptor_bytes.checked_add(
                crate::exact_buffer::layout_bytes::<IndexedAffineRow>(rows).map_err(|_| Error::Bounds)?)
                .ok_or(Error::Bounds)?;
            staging = staging.max(record_layouts(rows)?.staging);
            if proof {
                proof_bytes = proof_bytes.checked_add(backing_bytes(0, rows)?).ok_or(Error::Bounds)?;
                proofs_total = proofs_total.checked_add(rows as u64).ok_or(Error::Bounds)?;
            }
            rows_total = rows_total.checked_add(rows as u64).ok_or(Error::Bounds)?;
            allocation_attempts = allocation_attempts.checked_add(1 + usize::from(proof)).ok_or(Error::Bounds)?;
        }
        let checker_scratch_bytes = checking_scratch_bytes();
        // Finish's schedule array is not hidden in the proof scratch estimate.
        // It coexists with the digest state and bounded current hash records.
        let finish_scratch_bytes = std::mem::size_of::<[usize; MAX_NODES]>()
            + std::mem::size_of::<Sha256>() + 2 * std::mem::size_of::<Predicate>()
            + std::mem::size_of::<Rule>() + std::mem::size_of::<Node>()
            + std::mem::size_of::<WorkBudget>()
            + hash_cache_scratch_bytes() + cache_build_scratch_bytes();
        // Allocation seed/conversion controls overlap the resident borrowed
        // cache in prepare; cache-building returns are already dead here.
        let allocation_scratch_bytes = staging
            .checked_add(std::mem::size_of::<CachedDag<'static>>())
            .and_then(|n| n.checked_add(std::mem::size_of::<Audit>())).ok_or(Error::Bounds)?;
        let backing = descriptor_bytes.checked_add(proof_bytes).ok_or(Error::Bounds)?;
        let common = backing.checked_add(caller_descriptor_bytes)
            .and_then(|v| v.checked_add(std::mem::size_of::<PreparedMap>()))
            .and_then(|v| v.checked_add(std::mem::size_of::<Result<PreparedMap, Error>>()))
            .ok_or(Error::Bounds)?;
        let active = common.checked_add(checker_scratch_bytes.max(finish_scratch_bytes).max(allocation_scratch_bytes))
            .ok_or(Error::Bounds)?;
        let native_known_peak_bytes = active.checked_add(std::mem::size_of::<ResidentBuilder>())
            .and_then(|v| v.checked_add(std::mem::size_of::<Option<Program>>()))
            .and_then(|v| v.checked_add(std::mem::size_of::<Result<(), Error>>()))
            .ok_or(Error::Bounds)?;
        // The metadata-only local view overlaps prepare's checking/allocation,
        // not finish. Neither the resident workspace nor CachedDag owns it.
        let metadata_prepare = checker_scratch_bytes.max(allocation_scratch_bytes)
            .checked_add(attention_work::INDEXED_WORK_VIEW_BYTES).ok_or(Error::Bounds)?;
        let metadata_known_peak_bytes = common.checked_add(metadata_prepare.max(finish_scratch_bytes))
            .and_then(|v| v.checked_add(attention_work::IndexedWorkWorkspace::workspace_bytes()))
            .and_then(|v| v.checked_add(std::mem::size_of::<Result<WorkBudget, Error>>()))
            .ok_or(Error::Bounds)?;
        // At most two terms/row and two five-field predicates/proof. Each
        // SHA update/control block is bounded by 64 visits, separately from the
        // actual numeric checker walks. No credit enters native WorkBudget.
        let hash_visits_bound = (MAX_NODES as u64).checked_add(rows_total.checked_mul(8).ok_or(Error::Bounds)?)
            .and_then(|v| v.checked_add(proofs_total.checked_mul(16)?))
            .and_then(|v| v.checked_mul(64))
            .and_then(|v| v.checked_add(CACHE_VISITS_BOUND + MAX_NODES as u64 * 64)).ok_or(Error::Bounds)?;
        Ok(Self { descriptor_bytes, proof_bytes, caller_descriptor_bytes,
            allocation_scratch_bytes,
            pending_owner_bytes: std::mem::size_of::<Pending>(), checker_scratch_bytes, finish_scratch_bytes,
            native_known_peak_bytes, metadata_known_peak_bytes, allocation_attempts,
            finish_validation_visits: MAX_VALIDATION_VISITS, hash_visits_bound })
    }
}

pub(super) trait Dag {
    fn storage(&self) -> Option<&IndexedOwners>;
    fn extended(&self) -> bool;
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn backing_bytes(&self) -> usize;
    fn read(&self, value: Value) -> Result<Node, Error>;
    fn width(&self, value: Value) -> Result<u8, Error>;
    fn table(&self, value: Value) -> Result<&[IndexedAffineRow], Error>;
    fn rule(&self, value: Value) -> Option<Rule>;
    fn proofs(&self, value: Value) -> &[LaneProof];
}

impl Dag for Program {
    fn storage(&self) -> Option<&IndexedOwners> { self.extension.owners().ok() }
    fn extended(&self) -> bool { matches!(self.extension, Extension::Indexed(_)) }
    fn len(&self) -> usize { self.len }
    fn capacity(&self) -> usize { self.extension.capacity(self.lanes) }
    fn backing_bytes(&self) -> usize { self.extension.backing_bytes() }
    fn read(&self, v: Value) -> Result<Node, Error> {
        if usize::from(v.0) >= self.len { return Err(Error::Binding); }
        self.nodes[usize::from(v.0)].ok_or(Error::Binding)
    }
    fn width(&self, v: Value) -> Result<u8, Error> {
        self.read(v)?;
        Ok(self.bit_widths[usize::from(v.0)])
    }
    fn table(&self, v: Value) -> Result<&[IndexedAffineRow], Error> {
        self.read(v)?;
        match &self.extension {
        Extension::Absent | Extension::CheckedRowsIdentity(_) => {
            if matches!(self.read(v)?.instruction, Instruction::IndexedAffine { .. }) {
                return Err(Error::Phase);
            }
            Ok(&[])
        }
        Extension::Indexed(_) => self.extension.table(usize::from(v.0)),
        }
    }
    fn rule(&self, v: Value) -> Option<Rule> {
        match &self.extension { Extension::Absent | Extension::CheckedRowsIdentity(_) => None, Extension::Indexed(o) => o.rules[usize::from(v.0)] }
    }
    fn proofs(&self, v: Value) -> &[LaneProof] {
        match &self.extension { Extension::Absent | Extension::CheckedRowsIdentity(_) => &[], Extension::Indexed(o) => o.proofs(usize::from(v.0)) }
    }
}

// One borrowed cache per outer pass, never per lane/predicate. Every entry is
// still a reference into the actual physical owner; it grants no proof fact.
pub(super) struct RecordIndex<'a> { entries: [Option<&'a RecordLink>; MAX_NODES] }
pub(super) const fn hash_cache_scratch_bytes() -> usize {
    std::mem::size_of::<Option<RecordIndex<'static>>>() + std::mem::size_of::<Audit>()
}
pub(super) const fn cache_build_scratch_bytes() -> usize {
    // Named local index + distinct return + checked layouts/controls. No
    // stdlib Try internals or opaque machine callframe guesses.
    std::mem::size_of::<RecordIndex<'static>>()
        + std::mem::size_of::<Result<RecordIndex<'static>, Error>>()
        + std::mem::size_of::<linked_rows::Layouts>()
        + std::mem::size_of::<Result<linked_rows::Layouts, Error>>()
        + std::mem::size_of::<Option<&'static RecordLink>>()
        + std::mem::size_of::<&'static RecordLink>()
        + std::mem::size_of::<Node>() + std::mem::size_of::<Value>()
        + 6 * std::mem::size_of::<usize>()
        + std::mem::size_of::<&'static IndexedOwners>()
        + std::mem::size_of::<&'static dyn Dag>()
        + std::mem::size_of::<&'static mut Audit>()
}
pub(super) const CACHE_VISITS_BOUND: u64 = (MAX_NODES as u64 * 9 + 8) * 64;
pub(super) const RUNTIME_LOOKUP_VISITS_BOUND: u64 = MAX_NODES as u64 * MAX_NODES as u64 * 64;
// Original audit rows: Nested <=1488/row, Masked <=1328/row. Doubling
// their helper/control allowance covers the extra constant cached delegation
// (<3072/row); scalar/layout blocks fit 1024/node. Never multiply a predicate
// by chain length. These are source bounds, not a new fee or admission grant.
pub(super) const REPAIRED_VALIDATION_BOUND: u64 =
    MAX_NODES as u64 * (1024 * 3072 + 1024) + CACHE_VISITS_BOUND;
pub(super) const MAX_HASH_VISITS_BOUND: u64 =
    (MAX_NODES as u64 + MAX_NODES as u64 * CAPACITY as u64 * 24) * 64
        + CACHE_VISITS_BOUND + MAX_NODES as u64 * 64;
// Even summing the disjoint checker/hash/runtime lookup passes fits the SAME
// universal ceiling. The actual resource endpoint neither hashes nor executes.
const _: () = assert!(REPAIRED_VALIDATION_BOUND + MAX_HASH_VISITS_BOUND
    + RUNTIME_LOOKUP_VISITS_BOUND < MAX_VALIDATION_VISITS);
impl<'a> RecordIndex<'a> {
    pub(super) fn new(g: &'a dyn Dag, audit: &mut Audit) -> Result<Self, Error> {
        if g.len() == 0 || g.len() > MAX_NODES { return Err(Error::Bounds); }
        let owners = g.storage().ok_or(Error::Phase)?;
        let mut result = Self { entries: [None; MAX_NODES] };
        let mut link = owners.records.first();
        let mut backing = 0usize;
        audit.add(64 * (MAX_NODES + 8))?;
        while let Some(record) = link {
            audit.add(8 * 64)?; // bounded dispatch + typed layout + identity/proof/next
            let id = record.identity();
            let i = usize::from(id.0);
            if i == 0 || i >= g.len() || result.entries[i].is_some()
                || record.rows().is_empty() || record.copied() != record.rows().len() {
                return Err(Error::Binding);
            }
            let node = g.read(id)?;
            if !matches!(node.instruction, Instruction::IndexedAffine { table, output_lanes, .. }
                if table == id.0 && output_lanes == record.rows().len() && node.lanes == output_lanes) {
                return Err(Error::Binding);
            }
            let expected = record_layouts(record.rows().len())?;
            if record.allocation_bytes() != expected.backing { return Err(Error::Binding); }
            let proof_count = if matches!(g.rule(id), Some(Rule::NestedDifference { .. })) {
                record.rows().len()
            } else { 0 };
            if record.proofs().len() != proof_count { return Err(Error::Binding); }
            backing = backing.checked_add(record.allocation_bytes())
                .and_then(|n| n.checked_add(crate::exact_buffer::layout_bytes::<LaneProof>(proof_count).ok()?))
                .ok_or(Error::Bounds)?;
            result.entries[i] = Some(record);
            link = record.next().first();
        }
        if backing != owners.backing_bytes || owners.rules[g.len()..].iter().any(Option::is_some) {
            return Err(Error::Binding);
        }
        Ok(result)
    }
    pub(super) fn table(&self, i: usize) -> Result<&[IndexedAffineRow], Error> {
        self.entries.get(i).map(|record| record.map_or(&[][..], |r| r.rows())).ok_or(Error::Binding)
    }
    fn proofs(&self, i: usize) -> &[LaneProof] {
        self.entries[i].map_or(&[], |r| r.proofs())
    }
}
pub(super) struct CachedDag<'a> { graph: &'a dyn Dag, index: RecordIndex<'a> }
impl<'a> CachedDag<'a> {
    fn new(graph: &'a dyn Dag, audit: &mut Audit) -> Result<Self, Error> {
        Ok(Self { graph, index: RecordIndex::new(graph, audit)? })
    }
}
impl Dag for CachedDag<'_> {
    fn storage(&self) -> Option<&IndexedOwners> { self.graph.storage() }
    fn extended(&self) -> bool { self.graph.extended() }
    fn len(&self) -> usize { self.graph.len() }
    fn capacity(&self) -> usize { self.graph.capacity() }
    fn backing_bytes(&self) -> usize { self.graph.backing_bytes() }
    fn read(&self, value: Value) -> Result<Node, Error> { self.graph.read(value) }
    fn width(&self, value: Value) -> Result<u8, Error> { self.graph.width(value) }
    fn table(&self, value: Value) -> Result<&[IndexedAffineRow], Error> { self.index.table(usize::from(value.0)) }
    fn rule(&self, value: Value) -> Option<Rule> { self.graph.rule(value) }
    fn proofs(&self, value: Value) -> &[LaneProof] { self.index.proofs(usize::from(value.0)) }
}

fn before(g: &dyn Dag, value: Value, dependent: Value, audit: &mut Audit) -> Result<Node, Error> {
    audit.add(8)?;
    if value.0 >= dependent.0 { return Err(Error::Binding); }
    g.read(value)
}

fn indexed<'a>(g: &'a dyn Dag, value: Value, audit: &mut Audit)
    -> Result<(Value, &'a [IndexedAffineRow]), Error> {
    audit.add(16)?;
    let node = g.read(value)?;
    let Instruction::IndexedAffine { input, table, output_lanes } = node.instruction
        else { return Err(Error::Binding); };
    if table != value.0 || node.bit || output_lanes != node.lanes { return Err(Error::Binding); }
    before(g, input, value, audit)?;
    let rows = g.table(value)?;
    if rows.len() != output_lanes { return Err(Error::Shape); }
    Ok((input, rows))
}

fn predicate(g: &dyn Dag, packed: Value, lane: usize, dependent: Value, audit: &mut Audit)
    -> Result<Predicate, Error> {
    let b2a = before(g, packed, dependent, audit)?;
    audit.add(32)?;
    let Instruction::FromBit(sign) = b2a.instruction else { return Err(Error::Binding); };
    if b2a.bit || b2a.bounds != Interval::new(0, 1)? || lane >= b2a.lanes {
        return Err(Error::Binding);
    }
    let sign_node = before(g, sign, packed, audit)?;
    if !sign_node.bit || g.width(sign)? != 1 || sign_node.lanes != b2a.lanes
        || sign_node.bounds != Interval::new(0, 1)? {
        return Err(Error::Binding);
    }
    // Deliberately no CompactSign/LowBits/BitMap or arbitrary bit interval.
    let Instruction::Sign(offset) = sign_node.instruction else { return Err(Error::Binding); };
    let offset_node = before(g, offset, sign, audit)?;
    let (root, rows) = indexed(g, offset, audit)?;
    if offset_node.lanes != sign_node.lanes || g.rule(offset).is_some() {
        return Err(Error::Binding);
    }
    let source = before(g, root, offset, audit)?;
    let row = rows.get(lane).ok_or(Error::Shape)?;
    audit.add(24)?;
    if source.bit || row.len != 1 || row.terms[0].coefficient != 1
        || row.terms[1] != IndexedTerm::default() {
        return Err(Error::Binding);
    }
    // Validate the ACTUAL signed offset on the ACTUAL root domain. Exact MSB
    // is not permission to wrap an unwrapped signed comparison.
    audit.affine(std::slice::from_ref(row))?;
    IndexedAffine(std::slice::from_ref(row)).output_bounds(source.lanes, source.bounds)?;
    let threshold = i64::try_from(-i128::from(row.constant)).map_err(|_| Error::Bounds)?;
    Ok(Predicate { root, lane: row.terms[0].lane, offset, sign, b2a: packed,
        packed_lane: lane, threshold })
}

pub(super) fn derive_lane(g: &dyn Dag, packed: Value, row: &IndexedAffineRow, lane: usize,
    dependent: Value, outputs: usize, audit: &mut Audit) -> Result<LaneProof, Error> {
    audit.add(32)?;
    if row.constant != 0 || row.len != 2 || row.terms[0].coefficient != 1
        || row.terms[1].coefficient != -1 { return Err(Error::Binding); }
    let high = predicate(g, packed, row.terms[0].lane, dependent, audit)?;
    let low = predicate(g, packed, row.terms[1].lane, dependent, audit)?;
    if high.root != low.root || high.lane != low.lane || high.lane != lane
        || low.threshold > high.threshold || g.read(high.root)?.lanes != outputs {
        return Err(Error::Binding);
    }
    Ok(LaneProof { low, high })
}

pub(super) fn nested(g: &dyn Dag, packed: Value, rows: &[IndexedAffineRow], dependent: Value,
    audit: &mut Audit) -> Result<(), Error> {
    let mut root = None;
    for (lane, row) in rows.iter().enumerate() {
        let proof = derive_lane(g, packed, row, lane, dependent, rows.len(), audit)?;
        if root.replace(proof.low.root).is_some_and(|r| r != proof.low.root) {
            return Err(Error::Binding);
        }
    }
    Ok(())
}

fn check_nested(g: &dyn Dag, inside: Value, audit: &mut Audit) -> Result<(), Error> {
    audit.add(24)?;
    let Some(Rule::NestedDifference { packed }) = g.rule(inside) else { return Err(Error::Binding); };
    let (input, rows) = indexed(g, inside, audit)?;
    if input != packed || g.read(inside)?.bounds != Interval::new(0, 1)?
        || g.proofs(inside).len() != rows.len() { return Err(Error::Binding); }
    nested(g, packed, rows, inside, audit)?;
    for (lane, row) in rows.iter().enumerate() {
        let expected = derive_lane(g, packed, row, lane, inside, rows.len(), audit)?;
        audit.add(16)?;
        if g.proofs(inside)[lane] != expected { return Err(Error::Binding); }
    }
    Ok(())
}

pub(super) fn masked(g: &dyn Dag, original: Value, inside: Value, dependent: Value,
    audit: &mut Audit) -> Result<Interval, Error> {
    let cached = CachedDag::new(g, audit)?;
    masked_cached(&cached, original, inside, dependent, audit)
}
fn masked_cached(g: &dyn Dag, original: Value, inside: Value, dependent: Value,
    audit: &mut Audit) -> Result<Interval, Error> {
    let source = before(g, original, dependent, audit)?;
    let mask = before(g, inside, dependent, audit)?;
    if source.bit || mask.bit || source.lanes != mask.lanes { return Err(Error::Shape); }
    check_nested(g, inside, audit)?;
    // Independent ordinary raw-product check, using x's domain and mask [0,1],
    // before deriving any tighter result metadata. No certified product domain.
    audit.add(64)?;
    for x in [source.bounds.lower(), source.bounds.upper()] {
        for y in [mask.bounds.lower(), mask.bounds.upper()] {
            i64::try_from(i128::from(x).checked_mul(i128::from(y)).ok_or(Error::Bounds)?)
                .map_err(|_| Error::Bounds)?;
        }
    }
    let mut lo = 0;
    let mut hi = 0;
    for (lane, proof) in g.proofs(inside).iter().enumerate() {
        audit.add(32)?;
        if proof.low.root != original || proof.high.root != original
            || proof.low.lane != lane || proof.high.lane != lane { return Err(Error::Binding); }
        // i128 keeps t_high-1 representable even at i64::MIN.
        let lower = i128::from(source.bounds.lower()).max(i128::from(proof.low.threshold));
        let upper = i128::from(source.bounds.upper()).min(i128::from(proof.high.threshold) - 1);
        if lower <= upper {
            lo = lo.min(i64::try_from(lower).map_err(|_| Error::Bounds)?);
            hi = hi.max(i64::try_from(upper).map_err(|_| Error::Bounds)?);
        }
    }
    Interval::new(lo, hi)
}

pub(super) fn validate(g: &dyn Dag, audit: &mut Audit) -> Result<(), Error> {
    #[cfg(test)]
    CHECKER_ENTRIES.with(|n| n.set(n.get() + 1));
    if g.len() == 0 || g.len() > MAX_NODES { return Err(Error::Bounds); }
    if !g.extended() { return Ok(()); }
    if g.capacity() != CAPACITY || g.read(Value(0))?.lanes > MAX_TILE_LANES {
        return Err(Error::Bounds);
    }
    let cached = CachedDag::new(g, audit)?;
    let g: &dyn Dag = &cached;
    let mut backing = 0usize;
    for index in 0..g.len() {
        let value = Value(index as u8);
        let node = g.read(value)?;
        audit.add(24 + 8 * 64)?; // node plus bounded physical-layout dispatch
        if node.lanes == 0 || node.lanes > g.capacity() { return Err(Error::Shape); }
        backing = backing.checked_add(backing_bytes(g.table(value)?.len(), g.proofs(value).len())?)
            .ok_or(Error::Bounds)?;
        for input in node.instruction.inputs().into_iter().flatten() {
            before(g, input, value, audit)?;
        }
        if let Instruction::IndexedAffine { .. } = node.instruction {
            let (input, rows) = indexed(g, value, audit)?;
            let source = g.read(input)?;
            if source.bit || node.lanes > g.capacity() { return Err(Error::Binding); }
            audit.affine(rows)?;
            let generic = IndexedAffine(rows).output_bounds(source.lanes, source.bounds)?;
            if g.rule(value).is_none() && (node.bounds != generic || !g.proofs(value).is_empty()) {
                return Err(Error::Binding);
            }
        } else if !g.table(value)?.is_empty() { return Err(Error::Binding); }
        match g.rule(value) {
            Some(Rule::NestedDifference { .. }) => check_nested(g, value, audit)?,
            Some(Rule::MaskedOperand { original, inside }) => {
                if node.instruction != (Instruction::Product { left: original, right: inside,
                    rescale: false, certified: None }) || !g.proofs(value).is_empty()
                    || node.bit || node.bounds != masked_cached(g, original, inside, value, audit)? {
                    return Err(Error::Binding);
                }
            }
            None => {
                if !g.proofs(value).is_empty() { return Err(Error::Binding); }
            }
        }
    }
    if backing != g.backing_bytes() || audit.visits > MAX_VALIDATION_VISITS { return Err(Error::Bounds); }
    Ok(())
}

pub(super) fn hash_header(p: &Program, h: &mut Sha256) {
    match &p.extension {
    Extension::Absent => {},
    Extension::Indexed(o) => {
        h.update(EXTENSION);
        h.update((o.capacity as u64).to_le_bytes());
        h.update((p.lanes as u64).to_le_bytes());
    }
    Extension::CheckedRowsIdentity(identity) => {
        h.update(b"puma.checked-rows.v1");
        h.update(identity.rows.to_le_bytes());
        h.update(identity.width.to_le_bytes());
    }
    }
}

pub(super) fn hash_rule(p: &Program, cache: Option<&RecordIndex<'_>>, index: usize, h: &mut Sha256) {
    let Extension::Indexed(o) = &p.extension else { return; };
    let Some(rule) = o.rules[index] else { return; };
    h.update(EXTENSION);
    h.update([index as u8]);
    match rule {
        Rule::NestedDifference { packed } => h.update([0, packed.0]),
        Rule::MaskedOperand { original, inside } => h.update([1, original.0, inside.0]),
    }
    let proofs = cache.expect("indexed hash cache").proofs(index);
    h.update((proofs.len() as u64).to_le_bytes());
    for proof in proofs {
        for predicate in [proof.low, proof.high] {
            h.update([predicate.root.0, predicate.offset.0, predicate.sign.0, predicate.b2a.0]);
            h.update((predicate.lane as u64).to_le_bytes());
            h.update((predicate.packed_lane as u64).to_le_bytes());
            h.update(predicate.threshold.to_le_bytes());
        }
    }
}

pub(super) struct Pending {
    pub(super) rows: PendingRows,
    pub(super) proofs: ExactStorage<LaneProof>,
}

impl Pending {
    pub(super) const fn empty() -> Self { Self { rows: PendingRows::empty(), proofs: ExactStorage::Empty } }
    pub(super) fn is_empty(&self) -> bool {
        self.rows.record.is_none() && matches!(self.proofs, ExactStorage::Empty)
    }
    pub(super) fn ready(&self, nested: bool, node: Value) -> Result<(), Error> {
        let record = self.rows.record.as_ref().ok_or(Error::Phase)?;
        if !self.rows.sealed || record.identity() != node || record.copied() != record.rows().len()
            || record.next().first().is_some() || !record.proofs().is_empty() { return Err(Error::Phase); }
        match (&self.proofs, nested) {
            (ExactStorage::Ready(proofs), true) if proofs.len() == record.rows().len() => Ok(()),
            (ExactStorage::Empty, false) => Ok(()),
            _ => Err(Error::Phase),
        }
    }
    pub(super) fn publish(&mut self, owners: &mut IndexedOwners) {
        // Caller established ALL readiness/destination/overflow checks. These
        // are only moves and writes; no hook, allocation or fallible transfer.
        let mut record = self.rows.record.take().expect("ready descriptor");
        if let ExactStorage::Ready(proofs) = std::mem::replace(&mut self.proofs, ExactStorage::Empty) {
            *record.proofs_slot() = proofs;
        }
        owners.records.prepend(record);
        self.rows.sealed = false;
    }
}

pub(super) struct PendingRows { record: Option<RecordOwner>, sealed: bool }
impl PendingRows {
    const fn empty() -> Self { Self { record: None, sealed: false } }
    pub(super) fn len(&self) -> usize { self.record.as_ref().map_or(0, |r| r.copied()) }
    pub(super) fn extent(&self) -> usize { self.record.as_ref().map_or(0, |r| r.rows().len()) }
    fn acquire(&mut self, n: usize, node: Value) -> Result<(), Error> {
        if self.record.is_some() { return Err(Error::Phase); }
        // Concrete Copy row seed, not generic Default callback. Installation
        // immediately follows allocation/conversion, before every failure seam.
        self.record = Some(linked_rows::allocate(n, IndexedAffineRow::default(), node).map_err(|e|
            if e == crate::exact_buffer::Error::Allocation { Error::Allocation } else { Error::Bounds })?);
        Ok(())
    }
    fn push(&mut self, row: IndexedAffineRow) -> Result<(), Error> {
        if self.sealed { return Err(Error::Phase); }
        self.record.as_mut().ok_or(Error::Phase)?.push(row).map_err(|_| Error::Phase)
    }
    fn seal(&mut self) -> Result<(), Error> {
        if self.record.is_none() || self.len() != self.extent() || self.sealed { return Err(Error::Phase); }
        self.sealed = true;
        Ok(())
    }
}

/// Narrow extension; scalar-only work counters cannot issue these rules.
pub(super) trait IndexedSink: SoftmaxSink {
    fn indexed_affine(&mut self, input: Value, rows: &[IndexedAffineRow]) -> Result<Value, Error>;
    fn nested_difference(&mut self, packed: Value, rows: &[IndexedAffineRow]) -> Result<Value, Error>;
    fn masked_operand(&mut self, original: Value, inside: Value) -> Result<Value, Error>;
}

pub(super) struct PreparedMap {
    pub(super) value: Value,
    pub(super) node: Node,
    pub(super) width: u8,
    pub(super) bytes: usize,
    pub(super) visits: u64,
}

/// One shared construction/checking walk for native and work-only sinks. The
/// pending owners live in their original enclosing attempt on every exit.
pub(super) fn prepare_map(g: &dyn Dag, input: Value, rows: &[IndexedAffineRow],
    nested_rule: bool, pending: &mut Pending) -> Result<PreparedMap, Error> {
    if !g.extended() || g.capacity() != CAPACITY || !pending.is_empty() || g.len() >= MAX_NODES {
        return Err(Error::Phase);
    }
    let value = Value(g.len() as u8);
    let mut audit = Audit::default();
    let cached = CachedDag::new(g, &mut audit)?;
    let g: &dyn Dag = &cached;
    let source = before(g, input, value, &mut audit)?;
    if source.bit { return Err(Error::Binding); }
    let generic = IndexedAffine(rows).output_bounds(source.lanes, source.bounds)?;
    audit.affine(rows)?;
    if nested_rule { nested(g, input, rows, value, &mut audit)?; }
    let bounds = if nested_rule { Interval::new(0, 1)? } else { generic };
    let (value, node, width) = attention_work::checked_indexed_node_with(|v| g.read(v), |v| g.width(v),
        g.len(), g.capacity(), Instruction::IndexedAffine { input, table: value.0,
            output_lanes: rows.len() }, bounds, false)?;
    let bytes = backing_bytes(rows.len(), if nested_rule { rows.len() } else { 0 })?;
    #[cfg(test)]
    fault(Fault::AcquireRows)?;
    pending.rows.acquire(rows.len(), value)?;
    if nested_rule {
        #[cfg(test)]
        fault(Fault::AcquireProofs)?;
        pending.proofs.acquire(rows.len())?;
    }
    for (lane, row) in rows.iter().enumerate() {
        pending.rows.push(*row)?;
        #[cfg(test)]
        fault(Fault::Row)?;
        if nested_rule {
            pending.proofs.push(derive_lane(g, input, row, lane, value, rows.len(), &mut audit)?)?;
            #[cfg(test)]
            fault(Fault::Proof)?;
        }
    }
    pending.rows.seal()?;
    if nested_rule { pending.proofs.seal()?; }
    #[cfg(test)]
    fault(Fault::Publish)?;
    Ok(PreparedMap { value, node, width, bytes, visits: audit.visits })
}

impl Builder {
    pub(super) fn indexed(lanes: usize, input: Interval) -> Result<Self, Error> {
        let mut b = Self::new(lanes, input)?;
        b.0.extension = Extension::Indexed(IndexedOwners::empty());
        Ok(b)
    }

    pub(super) fn indexed_with_pending(&mut self, input: Value, rows: &[IndexedAffineRow],
        nested_rule: bool, pending: &mut Pending) -> Result<Value, Error> {
        let owners = self.0.extension.owners_mut()?;
        if owners.capacity != CAPACITY || !pending.is_empty() || self.0.len >= MAX_NODES {
            return Err(Error::Phase);
        }
        let index = self.0.len;
        if !owners.table(index).is_empty() || !owners.proofs(index).is_empty()
            || owners.rules[index].is_some() || self.0.nodes[index].is_some()
            || self.0.bit_widths[index] != 0 { return Err(Error::Phase); }
        let prepared = prepare_map(&self.0, input, rows, nested_rule, pending)?;
        let owners = self.0.extension.owners_mut()?;
        let total = owners.backing_bytes.checked_add(prepared.bytes).ok_or(Error::Bounds)?;
        let visits = owners.check_visits.checked_add(prepared.visits).ok_or(Error::Bounds)?;
        pending.ready(nested_rule, prepared.value)?;
        // All destination/readiness checks precede transfer. Remaining writes
        // are infallible; neither instruction nor proof can precede its backing.
        pending.publish(owners);
        owners.rules[index] = nested_rule.then_some(Rule::NestedDifference { packed: input });
        self.0.nodes[index] = Some(prepared.node);
        self.0.bit_widths[index] = prepared.width;
        owners.backing_bytes = total;
        owners.check_visits = visits;
        self.0.len += 1;
        Ok(prepared.value)
    }
}

impl IndexedSink for Builder {
    fn indexed_affine(&mut self, input: Value, rows: &[IndexedAffineRow]) -> Result<Value, Error> {
        self.indexed_with_pending(input, rows, false, &mut Pending::empty())
    }
    fn nested_difference(&mut self, input: Value, rows: &[IndexedAffineRow]) -> Result<Value, Error> {
        self.indexed_with_pending(input, rows, true, &mut Pending::empty())
    }
    fn masked_operand(&mut self, original: Value, inside: Value) -> Result<Value, Error> {
        if self.0.extension.owners_mut()?.capacity != CAPACITY { return Err(Error::Phase); }
        let value = Value(self.0.len as u8);
        let mut audit = Audit::default();
        let bounds = masked(&self.0, original, inside, value, &mut audit)?;
        let visits = self.0.extension.owners()?.check_visits.checked_add(audit.visits).ok_or(Error::Bounds)?;
        // Ordinary raw product is checked first and remains certified=None.
        let output = self.product(original, inside, false)?;
        self.0.nodes[usize::from(output.0)].as_mut().unwrap().bounds = bounds;
        let owners = self.0.extension.owners_mut()?;
        owners.rules[usize::from(output.0)] = Some(Rule::MaskedOperand { original, inside });
        owners.check_visits = visits;
        Ok(output)
    }
}

pub(super) fn backing_bytes(rows: usize, proofs: usize) -> Result<usize, Error> {
    (if rows == 0 { 0 } else { record_layouts(rows)?.backing })
        .checked_add(crate::exact_buffer::layout_bytes::<LaneProof>(proofs).map_err(|_| Error::Bounds)?)
        .ok_or(Error::Bounds)
}
fn record_layouts(rows: usize) -> Result<linked_rows::Layouts, Error> {
    linked_rows::layouts::<IndexedAffineRow, Value, LaneProof>(rows).map_err(|_| Error::Bounds)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fault { AcquireRows, AcquireProofs, Row, Proof, Publish,
    PanicAfterRow, PanicAfterProof, PanicBeforePublish }
#[cfg(test)]
thread_local! { static FAILURE: std::cell::Cell<Option<Fault>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub(super) fn arm_fault(stage: Fault) { FAILURE.with(|f| assert!(f.replace(Some(stage)).is_none())); }
#[cfg(test)]
fn fault(stage: Fault) -> Result<(), Error> {
    FAILURE.with(|f| {
        let unwind = matches!((f.get(), stage),
            (Some(Fault::PanicAfterRow), Fault::Row)
            | (Some(Fault::PanicAfterProof), Fault::Proof)
            | (Some(Fault::PanicBeforePublish), Fault::Publish));
        if unwind { f.set(None); panic!("injected indexed construction unwind"); }
        if f.get() == Some(stage) { f.set(None); Err(Error::Allocation) } else { Ok(()) }
    })
}
