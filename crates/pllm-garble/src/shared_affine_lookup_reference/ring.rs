//! Public-original-table specialization, not arbitrary private-partition FSS.
//! Shares one common input rotation through the existing universal DCF. No
//! framing, mask shares, PRG, products or transport live in this module. A private
//! native codec seam supports the enclosing consuming party transport owner.
use crate::{compact_dcf, interval_fss, shared_rescale_reference::Error};
use std::{borrow::Borrow, mem::size_of, sync::Arc};
use zeroize::Zeroizing;

pub(crate) const MAX_INTERVALS: usize = 513;

/// Immutable public table. Coefficients have fixed [constant, linear] ordering;
/// scale/descriptor provenance is the caller's responsibility. No implicit LUT
/// generation, interpolation, clipping, conversion or rescaling takes place.
pub(crate) struct RingTable {
    bits: u8,
    boundaries: Vec<u64>,
    coefficients: Vec<[u64; 2]>,
}

impl RingTable {
    /// Bounds this new owner's logical allocation. Borrowed caller input buffers
    /// remain caller-owned and must be accounted for separately during import.
    pub(crate) fn new(
        bits: u8,
        boundaries: &[u64],
        coefficients: &[[u64; 2]],
        allocation_limit: usize,
    ) -> Result<Self, Error> {
        Self::validate(bits, boundaries, coefficients)?;
        copy_public(bits, boundaries, coefficients, allocation_limit)
    }

    fn validate(bits: u8, boundaries: &[u64], coefficients: &[[u64; 2]]) -> Result<(), Error> {
        if !(1..=50).contains(&bits) {
            return Err(Error::Domain);
        }
        if coefficients.len() > MAX_INTERVALS {
            return Err(Error::Resource);
        }
        let n = 1u64 << bits;
        if coefficients.is_empty()
            || boundaries.len() != coefficients.len() + 1
            || boundaries.first() != Some(&0)
            || boundaries.last() != Some(&n)
            || boundaries.windows(2).any(|w| w[0] >= w[1])
            || coefficients.iter().flatten().any(|&v| v >= n)
        {
            return Err(Error::Domain);
        }
        Ok(())
    }

    /// Consume already charged public construction buffers without another copy.
    /// Validation is identical to the ordinary borrowed constructor.
    pub(crate) fn from_owned(
        bits: u8,
        boundaries: Vec<u64>,
        coefficients: Vec<[u64; 2]>,
        allocation_limit: usize,
    ) -> Result<Self, Error> {
        Self::validate(bits, &boundaries, &coefficients)?;
        let table = Self {
            bits,
            boundaries,
            coefficients,
        };
        if size_of::<Self>() + table.capacity_bytes() > allocation_limit {
            return Err(Error::Resource);
        }
        Ok(table)
    }

    pub(crate) fn bits(&self) -> u8 {
        self.bits
    }
    pub(crate) fn boundaries(&self) -> &[u64] {
        &self.boundaries
    }
    pub(crate) fn coefficients(&self) -> &[[u64; 2]] {
        &self.coefficients
    }
    pub(crate) fn intervals(&self) -> usize {
        self.coefficients.len()
    }

    pub(crate) fn resources(&self) -> Result<RingResources, Error> {
        let dcf = interval_fss::RingComparison::resources(self.bits, self.bits)?;
        Ok(account(self, [self.payload_bytes(); 2], [dcf; 2]))
    }

    pub(crate) fn issue_shared(
        self: &Arc<Self>,
        input_mask: u64,
        allocation_limit: usize,
    ) -> Result<SharedRingIssuance, Error> {
        issue_ring_shared(self, input_mask, allocation_limit)
    }

    fn payload_bytes(&self) -> usize {
        self.boundaries.len() * 8 + self.coefficients.len() * 16
    }

    fn capacity_bytes(&self) -> usize {
        self.boundaries.capacity() * 8 + self.coefficients.capacity() * 16
    }
}

/// No Arc/Clone or hidden sharing. Reserve every public-copy buffer fallibly;
/// actual Vec capacities, not encoded lengths, determine the ownership bound.
fn copy_public(
    bits: u8,
    boundaries: &[u64],
    coefficients: &[[u64; 2]],
    limit: usize,
) -> Result<RingTable, Error> {
    let expected = size_of::<RingTable>() + boundaries.len() * 8 + coefficients.len() * 16;
    if expected > limit {
        return Err(Error::Resource);
    }
    let mut b = Vec::new();
    let mut c = Vec::new();
    b.try_reserve_exact(boundaries.len())
        .map_err(|_| Error::Resource)?;
    c.try_reserve_exact(coefficients.len())
        .map_err(|_| Error::Resource)?;
    if size_of::<RingTable>() + b.capacity() * 8 + c.capacity() * 16 > limit {
        return Err(Error::Resource);
    }
    b.extend_from_slice(boundaries);
    c.extend_from_slice(coefficients);
    Ok(RingTable {
        bits,
        boundaries: b,
        coefficients: c,
    })
}

/// Whole scalar issuance/evaluation envelope, including the caller's owned source
/// RingTable plus TWO independent party copies. Payload counts are hypothetical
/// fixed-word encodings, not a raw codec. Capacities/owners are actual logical
/// Rust storage; scratch is conservative source-managed stack storage, not RSS,
/// allocator metadata, compiler spill bounds or caller/session reservations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RingResources {
    pub public_table_payload_bytes: usize,
    pub source_table_owner_bytes: usize,
    pub source_table_capacity_bytes: usize,
    pub party_table_capacity_bytes: [usize; 2],
    pub party_owner_bytes: usize,
    pub party_private_key_payload_bytes: [usize; 2],
    pub party_private_key_capacity_bytes: [usize; 2],
    pub issuance_scratch_bytes: usize,
    pub evaluation_scratch_bytes_per_party: usize,
    pub construction_peak_bytes: usize,
    pub evaluation_peak_bytes: usize,
    pub peak_logical_allocation_bytes: usize,
    pub dcf_evaluations_per_party: usize,
    pub dcf_path_levels_per_party: usize,
    pub coefficient_multiplications_per_party: usize,
}

fn outer_issuance_scratch() -> usize {
    // Corrected/fresh/peer base words, raw entropy, correction/delta temporaries,
    // returned party/table staging and two resource-report temporaries.
    128 + 2 * size_of::<RingParty>() + 2 * size_of::<RingTable>() + 2 * size_of::<RingResources>()
}

fn account(
    table: &RingTable,
    copies: [usize; 2],
    dcf: [compact_dcf::Resources; 2],
) -> RingResources {
    let source = size_of::<RingTable>() + table.capacity_bytes();
    let private = dcf.map(|c| c.key_capacity_bytes);
    let live = source
        + 2 * size_of::<RingParty>()
        + copies.iter().sum::<usize>()
        + private.iter().sum::<usize>();
    let issuance_scratch =
        outer_issuance_scratch() + dcf.iter().map(|c| c.issuance_scratch_bytes).max().unwrap();
    // DCF issuance scratch also conservatively covers its narrower evaluation
    // stack; no heap allocation occurs during lookup evaluation.
    let evaluation_scratch = 128 + dcf.iter().map(|c| c.issuance_scratch_bytes).max().unwrap();
    let construction = live + issuance_scratch;
    let evaluation = live + 2 * evaluation_scratch;
    RingResources {
        public_table_payload_bytes: table.payload_bytes(),
        source_table_owner_bytes: size_of::<RingTable>(),
        source_table_capacity_bytes: table.capacity_bytes(),
        party_table_capacity_bytes: copies,
        party_owner_bytes: size_of::<RingParty>(),
        party_private_key_payload_bytes: dcf.map(|c| c.encoded_key_bytes + 16),
        party_private_key_capacity_bytes: private,
        issuance_scratch_bytes: issuance_scratch,
        evaluation_scratch_bytes_per_party: evaluation_scratch,
        construction_peak_bytes: construction,
        evaluation_peak_bytes: evaluation,
        peak_logical_allocation_bytes: construction.max(evaluation),
        // Base eval(0) once, then every original interior boundary once.
        dcf_evaluations_per_party: table.intervals(),
        dcf_path_levels_per_party: table.intervals() * usize::from(table.bits),
        coefficient_multiplications_per_party: 2 * (table.intervals() - 1),
    }
}

/// Private one-evaluation owner: only original PUBLIC table, own DCF and own
/// randomized base. No input mask, translated order/boundaries or clear dealer
/// correction survives issuance. Dropping/cancelling erases the private owners.
pub(crate) struct RingParty<T = RingTable> {
    party: u8,
    table: T,
    comparison: interval_fss::RingComparison,
    base: Zeroizing<[u64; 2]>,
}

/// Fresh paired private material sharing only immutable public table storage.
pub(crate) type SharedRingIssuance = ([RingParty<Arc<RingTable>>; 2], RingResources);

impl<T: Borrow<RingTable>> RingParty<T> {
    /// Public table is supplied independently by receiving compiled owner. Only
    /// randomized base and this party's universal DCF cross private transport.
    pub(crate) fn write_material(&self, out: &mut Vec<u8>) {
        for value in self.base.iter() {
            out.extend_from_slice(&value.to_le_bytes());
        }
        self.comparison.write_material(out);
    }
}

impl RingParty {
    pub(crate) fn read_material(
        input: &mut crate::fusefss_backend::wire::Reader<'_>,
        party: u8,
        table: &RingTable,
        budget: &crate::fusefss_backend::Budget,
    ) -> Result<Self, crate::fusefss_backend::Error> {
        use crate::fusefss_backend::allocate;
        let (base, comparison) = read_private(input, party, table, budget)?;
        let mut boundaries = allocate(table.boundaries.len(), budget)?;
        let mut coefficients = allocate(table.coefficients.len(), budget)?;
        boundaries.extend_from_slice(&table.boundaries);
        coefficients.extend_from_slice(&table.coefficients);
        Ok(Self {
            party,
            table: RingTable {
                bits: table.bits,
                boundaries,
                coefficients,
            },
            comparison,
            base,
        })
    }
}

impl RingParty<Arc<RingTable>> {
    /// Import private material against the receiving process's canonical public
    /// owner. No dealer pointer, translated boundary, or private base is shared.
    pub(crate) fn read_shared_material(
        input: &mut crate::fusefss_backend::wire::Reader<'_>,
        party: u8,
        table: Arc<RingTable>,
        budget: &crate::fusefss_backend::Budget,
    ) -> Result<Self, crate::fusefss_backend::Error> {
        let (base, comparison) = read_private(input, party, &table, budget)?;
        Ok(Self {
            party,
            table,
            comparison,
            base,
        })
    }
}

fn read_private(
    input: &mut crate::fusefss_backend::wire::Reader<'_>,
    party: u8,
    table: &RingTable,
    budget: &crate::fusefss_backend::Budget,
) -> Result<(Zeroizing<[u64; 2]>, interval_fss::RingComparison), crate::fusefss_backend::Error> {
    use crate::fusefss_backend::{primitive_error, Error as E};
    if party > 1 {
        return Err(E::Domain);
    }
    let base = Zeroizing::new([input.word(table.bits)?, input.word(table.bits)?]);
    let bytes = interval_fss::RingComparison::resources(table.bits, table.bits)
        .map_err(primitive_error)?
        .encoded_key_bytes;
    let comparison = interval_fss::RingComparison::read_material(
        input.take(bytes)?,
        table.bits,
        table.bits,
        budget,
    )?;
    Ok((base, comparison))
}

impl<T: Borrow<RingTable>> RingParty<T> {
    pub(crate) fn evaluate(self, opened: u64) -> Result<Zeroizing<[u64; 2]>, Error> {
        let table = self.table.borrow();
        let n = 1u64 << table.bits;
        if opened >= n {
            return Err(Error::Domain);
        }
        let mask = n - 1;
        let zero = Zeroizing::new(self.comparison.eval(self.party, opened, 0)?);
        let mut result = Zeroizing::new(*self.base);
        for (j, pair) in table.coefficients.windows(2).enumerate() {
            let indicator = Zeroizing::new(
                self.comparison
                    .eval(self.party, opened, table.boundaries[j + 1])?
                    .wrapping_sub(*zero)
                    & mask,
            );
            for coordinate in 0..2 {
                let delta = pair[0][coordinate].wrapping_sub(pair[1][coordinate]) & mask;
                result[coordinate] =
                    result[coordinate].wrapping_add(delta.wrapping_mul(*indicator)) & mask;
            }
        }
        Ok(result)
    }

    /// Consumes both public-copy and private-key owners; no reusable key remains.
    #[cfg(test)]
    pub(crate) fn cancel(self) {}
}

/// Dealer-only scalar issuance. Caller supplies fresh canonical input mask and
/// separately distributes its shares. This helper never stores those shares.
/// Limit covers original RingTable, both parties, copies and conservative scratch
/// for construction OR simultaneous evaluation. No process-wide lease is acquired.
pub(crate) fn issue_ring(
    table: &RingTable,
    input_mask: u64,
    allocation_limit: usize,
) -> Result<([RingParty; 2], RingResources), Error> {
    issue_inner(
        table,
        input_mask,
        allocation_limit,
        interval_fss::RingComparison::issue,
        |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
    )
}

fn issue_inner(
    table: &RingTable,
    input_mask: u64,
    allocation_limit: usize,
    issue_dcf: impl FnOnce(u8, u8, u64, usize) -> Result<[interval_fss::RingComparison; 2], Error>,
    random: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<([RingParty; 2], RingResources), Error> {
    issue_with_tables(
        table,
        input_mask,
        allocation_limit,
        || {
            let a = copy_public(
                table.bits,
                &table.boundaries,
                &table.coefficients,
                allocation_limit,
            )?;
            let b = copy_public(
                table.bits,
                &table.boundaries,
                &table.coefficients,
                allocation_limit,
            )?;
            let copies = [a.capacity_bytes(), b.capacity_bytes()];
            Ok(([a, b], copies))
        },
        issue_dcf,
        random,
    )
}

/// Opt-in public ownership only. The caller retains and charges the canonical
/// Arc backing independently through the last evaluator. Private issuance is
/// the same algorithm and deliberately keeps the legacy conservative envelope.
pub(crate) fn issue_ring_shared(
    table: &Arc<RingTable>,
    input_mask: u64,
    allocation_limit: usize,
) -> Result<SharedRingIssuance, Error> {
    issue_with_tables(
        table,
        input_mask,
        allocation_limit,
        || Ok(([Arc::clone(table), Arc::clone(table)], [0; 2])),
        interval_fss::RingComparison::issue,
        |bytes| getrandom::fill(bytes).map_err(|_| Error::Randomness),
    )
}

fn issue_with_tables<T: Borrow<RingTable>>(
    table: &RingTable,
    input_mask: u64,
    allocation_limit: usize,
    tables: impl FnOnce() -> Result<([T; 2], [usize; 2]), Error>,
    issue_dcf: impl FnOnce(u8, u8, u64, usize) -> Result<[interval_fss::RingComparison; 2], Error>,
    mut random: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<([RingParty<T>; 2], RingResources), Error> {
    let input_mask = Zeroizing::new(input_mask);
    let n = 1u64 << table.bits;
    if *input_mask >= n {
        return Err(Error::Domain);
    }
    let predicted = table.resources()?;
    if predicted.peak_logical_allocation_bytes > allocation_limit {
        return Err(Error::Resource);
    }

    // Public copies (or clones of the already charged immutable Arc) are
    // acquired BEFORE DCF issuance. DCF reserves both correction owners before RNG.
    // No dynamic allocation follows that first entropy request in this module.
    let ([a_table, b_table], copies) = tables()?;
    let dcf_cost = interval_fss::RingComparison::resources(table.bits, table.bits)?;
    let actual_tables = account(table, copies, [dcf_cost; 2]);
    if actual_tables.peak_logical_allocation_bytes > allocation_limit {
        return Err(Error::Resource);
    }
    let non_dcf = size_of::<RingTable>()
        + table.capacity_bytes()
        + copies.iter().sum::<usize>()
        + 2 * (size_of::<RingParty>() - dcf_cost.key_owner_bytes)
        + outer_issuance_scratch();
    // Also reserve simultaneous evaluator scratch when deriving DCF's budget.
    let extra_eval = actual_tables
        .evaluation_peak_bytes
        .saturating_sub(actual_tables.construction_peak_bytes);
    let [a_key, b_key] = issue_dcf(
        table.bits,
        table.bits,
        *input_mask,
        allocation_limit - non_dcf - extra_eval,
    )?;
    let costs = [a_key.allocation(), b_key.allocation()];
    let cost = account(table, copies, costs);
    // Actual DCF capacities were checked against the same remaining envelope
    // before RNG by its issuer. No post-entropy allocation or admission change.
    debug_assert!(cost.peak_logical_allocation_bytes <= allocation_limit);

    let mask = n - 1;
    let mut corrected = Zeroizing::new(*table.coefficients.last().unwrap());
    for (j, pair) in table.coefficients.windows(2).enumerate() {
        let correction = Zeroizing::new(interval_fss::RingComparison::lt_correction(
            table.bits,
            table.bits,
            *input_mask,
            table.boundaries[j + 1],
        )?);
        for coordinate in 0..2 {
            let delta = pair[0][coordinate].wrapping_sub(pair[1][coordinate]) & mask;
            corrected[coordinate] =
                corrected[coordinate].wrapping_add(delta.wrapping_mul(*correction)) & mask;
        }
    }
    let mut raw = Zeroizing::new([0; 16]);
    random(&mut *raw)?;
    let left = Zeroizing::new([
        u64::from_le_bytes(raw[..8].try_into().unwrap()) & mask,
        u64::from_le_bytes(raw[8..].try_into().unwrap()) & mask,
    ]);
    let right = Zeroizing::new([
        corrected[0].wrapping_sub(left[0]) & mask,
        corrected[1].wrapping_sub(left[1]) & mask,
    ]);
    Ok((
        [
            RingParty {
                party: 0,
                table: a_table,
                comparison: a_key,
                base: left,
            },
            RingParty {
                party: 1,
                table: b_table,
                comparison: b_key,
                base: right,
            },
        ],
        cost,
    ))
}

#[cfg(test)]
mod tests;
