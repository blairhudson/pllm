//! Bounded request-local immutable native source tables. A hit clones only Arc;
//! no entry generation, table payload copy or content hash occurs on that path.
//! Validated original-public lowered chunks share a separate leased backing.
//! Fresh preprocessing, masks and randomized bases are deliberately NOT cached.
use super::super::{allocate, lookup, Budget, Error, Lease};
use super::source_tables;
use std::{cell::RefCell, mem::size_of, sync::Arc};

pub const MAX_ENTRIES: usize = super::MAX_TABLES;
pub const MAX_RETAINED_BYTES: usize = super::MAX_PLAN_BYTES;

/// Closed source formulas only, never caller-provided values or source pointers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Silu,
    Exp0,
    Exp1,
    Reciprocal { k: usize },
    Rsqrt { k: usize },
}
impl Kind {
    fn entries(self) -> Result<usize, Error> {
        match self {
            Self::Silu => Ok(1024),
            Self::Exp0 | Self::Exp1 => Ok(256),
            Self::Reciprocal { k } => Ok(1usize << source_tables::reciprocal_q(k)?),
            Self::Rsqrt { k } if (1..=4096).contains(&k) => Ok(8192),
            _ => Err(Error::Domain),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_entries: usize,
    /// Native retained Table/Arc leases; fixed cache metadata is charged separately.
    pub max_retained_bytes: usize,
}

/// Exact content/geometry/source commitment. Key comparison is structural (no
/// hash-map collisions). Digest collision resistance is the existing Table hash's
/// assumption. This commitment is not publisher authentication or a range proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    pub namespace: [u8; 32],
    pub numeric_revision: [u8; 32],
    pub source_digest: [u8; 32],
    pub kind: Kind,
    pub bits: u8,
    pub entries: usize,
    pub table_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub references: usize,
    pub hits: usize,
    pub misses: usize,
    pub table_constructions: usize,
    pub table_hashes: usize,
    pub table_hash_bytes: usize,
    pub table_payload_copy_bytes: usize,
    /// Original public partitions constructed once, never randomized party data.
    pub lowered_chunk_constructions: usize,
    pub lowered_payload_bytes: usize,
    /// Completed lowered headers, Arc counters and stored handles, including
    /// failed attempts. Separate from payload copies and live backing leases.
    pub lowered_construction_metadata_bytes: usize,
    pub lowered_retained_bytes: usize,
    pub generated_words: usize,
    /// Cache-held shared owners, also possibly held by live Plans. Not exclusive
    /// memory: raw owners follow cache/Plan handles; lowered backing follows
    /// the last cache/Plan/issued/imported party handle independently.
    pub retained_bytes: usize,
    pub retained_entries: usize,
    pub metadata_reservation_bytes: usize,
    pub numeric_revision_hash_bytes: usize,
}

pub(super) struct SharedTable<'a> {
    pub(super) table: lookup::Table<'a>,
    _lease: Lease<'a>,
}
pub(super) fn shared_metadata_bytes() -> usize {
    // Table's own lease already charges its inline metadata. Add Arc counters,
    // the extra lease, and any SharedTable alignment; Arc handles live in charged
    // cache slots or in Builder's existing bounded owner slack.
    size_of::<SharedTable<'_>>() - size_of::<lookup::Table<'_>>() + 2 * size_of::<usize>()
}
struct Entry<'a> {
    identity: Identity,
    table: Arc<SharedTable<'a>>,
}
struct State<'a> {
    entries: Vec<Entry<'a>>,
    stats: Stats,
}

/// Borrow only the external Budget, NOT the cache, from returned Plans. The
/// request owner may retire this cache before any of its Plans/views retire.
/// Not global; interior mutation is serialized by a non-reentrant RefCell.
pub struct SourceTableCache<'a> {
    namespace: [u8; 32],
    numeric_revision: [u8; 32],
    limits: Limits,
    state: RefCell<State<'a>>,
    lease: Lease<'a>,
}
impl<'a> SourceTableCache<'a> {
    pub fn new(namespace: [u8; 32], limits: Limits, budget: &'a Budget) -> Result<Self, Error> {
        if namespace == [0; 32]
            || limits.max_entries == 0
            || limits.max_entries > MAX_ENTRIES
            || limits.max_retained_bytes == 0
            || limits.max_retained_bytes > MAX_RETAINED_BYTES
        {
            return Err(Error::Domain);
        }
        let metadata = size_of::<Self>() + limits.max_entries * size_of::<Entry<'_>>() + 4096;
        let lease = budget.reserve(metadata)?;
        let entries = allocate(limits.max_entries, budget)?;
        // Fixed source lock/numeric revision is hashed once per cache, not per
        // gate or reference. Formula changes must revise this source commitment.
        let numeric_revision = source_tables::source_digest();
        Ok(Self {
            namespace,
            numeric_revision,
            limits,
            state: RefCell::new(State {
                entries,
                stats: Stats {
                    metadata_reservation_bytes: metadata,
                    numeric_revision_hash_bytes: source_tables::source_lock_bytes(),
                    ..Stats::default()
                },
            }),
            lease,
        })
    }
    pub fn budget(&self) -> &'a Budget {
        self.lease.budget
    }
    pub fn stats(&self) -> Stats {
        self.state.borrow().stats
    }
    pub fn namespace(&self) -> [u8; 32] {
        self.namespace
    }
    pub fn numeric_revision(&self) -> [u8; 32] {
        self.numeric_revision
    }

    /// Prefetch or inspect one trusted fixed table. Acquisition counts as a
    /// reference, even if caller only requests identity and immediately drops it.
    pub fn identity(&self, kind: Kind, source_digest: [u8; 32]) -> Result<Identity, Error> {
        let table = self.get(kind, source_digest)?;
        Ok(Identity {
            namespace: self.namespace,
            numeric_revision: self.numeric_revision,
            source_digest,
            kind,
            bits: table.table.bits(),
            entries: table.table.entries(),
            table_digest: table.table.digest(),
        })
    }

    /// Exact substitution check against an already resident canonical owner. No
    /// generation/hash/allocation and no caller metadata can replace that owner.
    pub fn validate(&self, expected: Identity) -> Result<(), Error> {
        if expected.namespace != self.namespace
            || expected.numeric_revision != self.numeric_revision
        {
            return Err(Error::Domain);
        }
        let state = self.state.borrow();
        let entry = state
            .entries
            .iter()
            .find(|e| e.identity == expected)
            .ok_or(Error::Domain)?;
        self.check(entry, expected.kind, expected.source_digest)
    }

    fn check(&self, entry: &Entry<'_>, kind: Kind, source: [u8; 32]) -> Result<(), Error> {
        let id = entry.identity;
        if id.namespace != self.namespace
            || id.numeric_revision != self.numeric_revision
            || id.kind != kind
            || id.source_digest != source
            || id.bits != 50
            || id.entries != kind.entries()?
            || id.table_digest != entry.table.table.digest()
            || entry.table.table.bits() != id.bits
            || entry.table.table.entries() != id.entries
            || entry.table.table.source_digest() != source
        {
            return Err(Error::Domain);
        }
        Ok(())
    }

    pub(super) fn get(&self, kind: Kind, source: [u8; 32]) -> Result<Arc<SharedTable<'a>>, Error> {
        if source == [0; 32] {
            return Err(Error::Domain);
        }
        let count = kind.entries()?;
        let mut state = self.state.borrow_mut();
        if let Some(index) = state
            .entries
            .iter()
            .position(|e| e.identity.kind == kind && e.identity.source_digest == source)
        {
            self.check(&state.entries[index], kind, source)?;
            let owner = Arc::clone(&state.entries[index].table);
            state.stats.hits = state.stats.hits.checked_add(1).ok_or(Error::Resource)?;
            state.stats.references = state
                .stats
                .references
                .checked_add(1)
                .ok_or(Error::Resource)?;
            return Ok(owner);
        }
        state.stats.misses = state.stats.misses.checked_add(1).ok_or(Error::Resource)?;
        let mut retained = size_of::<lookup::Table<'_>>() + count * 8 + shared_metadata_bytes();
        if state.entries.len() == self.limits.max_entries
            || state
                .stats
                .retained_bytes
                .checked_add(retained)
                .ok_or(Error::Resource)?
                > self.limits.max_retained_bytes
        {
            return Err(Error::Resource);
        }
        let budget = self.budget();
        // Meter Arc metadata before its allocation; the extra lease follows the
        // immutable owner, independently of cache/Plan retirement order.
        let lease = budget.reserve(shared_metadata_bytes())?;
        budget.allocation()?;
        let table = match kind {
            Kind::Silu => {
                lookup::Table::new_zero_extended(50, source_tables::silu(), source, budget)?
            }
            Kind::Exp0 => {
                lookup::Table::new_zero_extended(50, source_tables::exp0(), source, budget)?
            }
            Kind::Exp1 => {
                lookup::Table::new_zero_extended(50, source_tables::exp1(), source, budget)?
            }
            Kind::Reciprocal { k } => {
                let values = source_tables::reciprocal(k, budget)?;
                state.stats.generated_words += values.len();
                lookup::Table::new_zero_extended(50, &values, source, budget)?
            }
            Kind::Rsqrt { k } => {
                let values = source_tables::rsqrt(k, budget)?;
                state.stats.generated_words += values.len();
                lookup::Table::new_zero_extended(50, &values, source, budget)?
            }
        };
        // These copies/hashes already happened, even if lowering or admission
        // fails later. Failed attempts never publish retained owners or hits.
        state.stats.table_constructions += 1;
        state.stats.table_hashes += 1;
        state.stats.table_hash_bytes += lookup::Table::construction_hash_bytes(count);
        state.stats.table_payload_copy_bytes += count * 8;
        let lowered_retained = table.lowered_reservation_bytes();
        retained = retained
            .checked_add(lowered_retained)
            .ok_or(Error::Resource)?;
        if state
            .stats
            .retained_bytes
            .checked_add(retained)
            .ok_or(Error::Resource)?
            > self.limits.max_retained_bytes
        {
            return Err(Error::Resource);
        }
        let table = table.with_shared_chunks_counted(|payload, metadata| {
            state.stats.lowered_chunk_constructions += usize::from(payload != 0);
            state.stats.lowered_payload_bytes += payload;
            state.stats.table_payload_copy_bytes += payload;
            state.stats.lowered_construction_metadata_bytes += metadata;
        })?;
        // Construction/hash are completed actual work, counted separately from
        // per-reference public handles and unchanged fresh private material.
        state.stats.lowered_retained_bytes += lowered_retained;
        let identity = Identity {
            namespace: self.namespace,
            numeric_revision: self.numeric_revision,
            source_digest: source,
            kind,
            bits: 50,
            entries: count,
            table_digest: table.digest(),
        };
        let owner = Arc::new(SharedTable {
            table,
            _lease: lease,
        });
        state.entries.push(Entry {
            identity,
            table: Arc::clone(&owner),
        });
        state.stats.references += 1;
        state.stats.retained_entries += 1;
        state.stats.retained_bytes += retained;
        Ok(owner)
    }
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
