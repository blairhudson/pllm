//! Public unsigned coefficient-digit embedding and three-row RNS contraction.
//!
//! No ciphertext/producer authority, keys, automorphism,
//! backend activation, parameter adoption, or encrypted orchestration. Inputs
//! may represent public c = sigma(c1), but this module does not compute sigma.
//! Variable-time clear algebra only. Existing legacy degree gates are unchanged;
//! degree 8192 requires explicitly Public8192-bounded NTT descriptors.
//!
//! Source spectra have shape [3][N], rows [3][2][4][N], embedded digits
//! [3][4][N], and result [2][4][N], all flattened in that order. Components
//! are (b,a); targets are ordered [q0,q1,q2,p]. Source inverse NTT precedes
//! coefficient-canonical UNSIGNED lifting into every target and target forward
//! NTT. Copying source NTT ordinates across moduli is not this operation.

use crate::coefficient_layout::modular::mul;
use crate::coefficient_ntt::{LogicalOperations, NegacyclicNtt, Operation};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    ContextMismatch,
    BufferLength,
    NonCanonical,
    ArithmeticCapacity,
    WorkLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Evaluation {
    Embed,
    Contract,
}

/// Componentwise conservative logical event bounds, not scalar credits, a
/// resource ledger, physical memory, time, or a complete runtime price. The NTT
/// owner's nested-category semantics apply. Reductions count unsigned coefficient
/// `% target_modulus` events separately from modular multiplications. Checks
/// include preflight, repeated NTT validation, and the finite comparison census
/// documented on `RnsContraction::work`. These are logical predicates, not CPU
/// instructions: loop/iterator termination, index/bounds machinery, boolean
/// short-circuit/branch instructions, unchecked arithmetic, and library-internal
/// implementation details are excluded. Capacity adapters and Result tags are
/// explicitly included below rather than treating them as excluded control.
/// Descriptor construction/primality/root setup is outside this operation quote.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Work {
    pub events: LogicalOperations,
    pub coefficient_reductions: u64,
}

macro_rules! event_fields {
    ($action:ident, $left:expr, $right:expr) => {
        $action!($left, $right;
            checks, residue_checks, moves, swaps, modular_additions,
            modular_subtractions, modular_multiplications, exponentiations,
            exponent_steps, primality_rounds)
    };
}

impl Work {
    fn add_ntt(&mut self, other: LogicalOperations) -> Result<(), Error> {
        macro_rules! add {
            ($left:expr, $right:expr; $($field:ident),*) => {
                $( $left.$field = $left.$field.checked_add($right.$field)
                    .ok_or(Error::ArithmeticCapacity)?; )*
            };
        }
        event_fields!(add, self.events, other);
        Ok(())
    }

    /// No scalarization: each event category must fit its own caller ceiling.
    /// The operation quote includes its ONE preflight call here (10+1 fields),
    /// not arbitrary caller invocations of this method.
    pub fn fits(self, ceiling: Self) -> bool {
        macro_rules! fits {
            ($left:expr, $right:expr; $($field:ident),*) => {
                true $( && $left.$field <= $right.$field )*
            };
        }
        event_fields!(fits, self.events, ceiling.events)
            && self.coefficient_reductions <= ceiling.coefficient_reductions
    }
}

/// Required simultaneously-live u64 payload prefixes, not physical peak memory.
/// Caller-provided excess scratch tails and descriptor storage are excluded.
/// No owned buffers, allocations, retained data, or hidden transaction copies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Storage {
    pub source_words: usize,
    pub row_words: usize,
    pub output_words: usize,
    pub scratch_words: usize,
    pub total_words: usize,
    pub total_bytes: usize,
}

/// Public source spectra plus their asserted ordered transform contexts.
/// This binding checks descriptor identity, not truth/provenance of supplied data.
pub struct SourceSpectra<'a> {
    pub contexts: &'a [NegacyclicNtt; 3],
    pub values: &'a [u64],
}

/// Public synthetic row spectra, component order (b,a), target order [q0,q1,q2,p].
pub struct RowSpectra<'a> {
    pub contexts: &'a [NegacyclicNtt; 4],
    pub values: &'a [u64],
}

/// Borrowed immutable validated NTT plans. Source order is caller-defined, not
/// numerical sorting. All four target primes must be distinct and same-degree;
/// first three target descriptors must equal corresponding source descriptors,
/// INCLUDING root identity, not merely ring modulus/degree.
pub struct RnsContraction<'a> {
    sources: &'a [NegacyclicNtt; 3],
    targets: &'a [NegacyclicNtt; 4],
}

impl<'a> RnsContraction<'a> {
    pub fn new(
        sources: &'a [NegacyclicNtt; 3],
        targets: &'a [NegacyclicNtt; 4],
    ) -> Result<Self, Error> {
        let degree = sources[0].ring().degree();
        for i in 0..4 {
            if targets[i].ring().degree() != degree
                || (i < 3 && sources[i] != targets[i])
            {
                return Err(Error::ContextMismatch);
            }
            for j in 0..i {
                if targets[i].ring().modulus() == targets[j].ring().modulus() {
                    return Err(Error::ContextMismatch);
                }
            }
        }
        Ok(Self { sources, targets })
    }

    pub fn degree(&self) -> usize {
        self.sources[0].ring().degree()
    }

    pub fn storage(&self, evaluation: Evaluation) -> Result<Storage, Error> {
        let n = self.degree();
        let words = |count: usize| n.checked_mul(count).ok_or(Error::ArithmeticCapacity);
        let source_words = words(3)?;
        let (row_words, output_words, scratch_words) = match evaluation {
            Evaluation::Embed => (0, words(12)?, words(2)?),
            Evaluation::Contract => (words(24)?, words(8)?, words(3)?),
        };
        let total_words = source_words.checked_add(row_words)
            .and_then(|v| v.checked_add(output_words))
            .and_then(|v| v.checked_add(scratch_words))
            .ok_or(Error::ArithmeticCapacity)?;
        let total_bytes = total_words.checked_mul(core::mem::size_of::<u64>())
            .ok_or(Error::ArithmeticCapacity)?;
        Ok(Storage {
            source_words, row_words, output_words, scratch_words, total_words, total_bytes,
        })
    }

    /// Per embed/contract call, including the single storage/work computation
    /// inside preflight, not earlier caller queries or `Self::new` setup.
    /// The unchanged 1024 non-residue check envelope has this finite census;
    /// contract maximizes every term (embed and early refusal are bounded by it):
    ///
    /// | Predicate category | Maximum |
    /// | --- | ---: |
    /// | Descriptor scalar equality: 7 * (2 ring + 3 scalar + 2*13 roots) | 217 |
    /// | Input/row/output/scratch length comparisons | 4 |
    /// | One preflight Work::fits: 10 event fields + reductions | 11 |
    /// | Checked capacity outcomes: storage 8 + work 157 | 165 |
    /// | Capacity Option adapters / conversion map_err: 8 + 157 | 165 |
    /// | Result tags: storage 6 + work 157 + helper 15 + preflight 3 + NTT 15 | 196 |
    /// | Aggregate context/length/work/noncanonical refusal predicates | 4 |
    /// | Row Option shape dispatches 4 + storage/work Evaluation selections 2 | 6 |
    /// | NTT quote selections: 15 * (operation match + N==1 + inverse selection) | 45 |
    /// | Total (211 unused checks remain below 1024) | 813 |
    ///
    /// Capacity census from actual calls: storage has four word multiplications,
    /// three total-word additions, one byte multiplication = 8. Its four word
    /// ok_or adapters, two and_then adapters, total ok_or and byte ok_or = 8;
    /// six internal `?` tags propagate four word Results plus total and bytes.
    /// Work has one usize->u64 conversion, FIVE scaled multiplications (residues,
    /// moves, additions, products, reductions), one check-envelope addition, and
    /// 15 add_ntt calls * TEN checked field additions = 157 capacity outcomes.
    /// These have 156 ok_or adapters plus one conversion map_err, and 157
    /// internal `?` tags. The helper calls add 15 Result tags; preflight adds
    /// storage?, work?, then its caller's preflight? = 3. Fifteen NTT expect
    /// tags are counted separately; validated inputs make their failure arms
    /// unreachable. No panic rollback guarantee is provided.
    ///
    /// Own residue comparisons (3N / 27N) and repeated NTT validation
    /// 15*(N+2) are separately added, NEVER charged again to this 813 census.
    /// NTT execution retains its owner's bundled check/correction semantics.
    /// Per-row sum corrections are included in modular_additions, not checks.
    /// RnsContraction::new is not called here; its checks are not operation work.
    pub fn work(&self, evaluation: Evaluation) -> Result<Work, Error> {
        let n = u64::try_from(self.degree()).map_err(|_| Error::ArithmeticCapacity)?;
        let scaled = |count: u64| n.checked_mul(count).ok_or(Error::ArithmeticCapacity);
        let contract = evaluation == Evaluation::Contract;
        let residues = scaled(if contract { 27 } else { 3 })?;
        let mut work = Work {
            events: LogicalOperations {
                // Proven 813-predicate census above fits unchanged 1024 envelope.
                // This is not a scalar tariff or a complete CPU-work inventory.
                checks: residues.checked_add(1024).ok_or(Error::ArithmeticCapacity)?,
                residue_checks: residues,
                // Lift stores, plus result initialization and 24N accumulations.
                moves: scaled(if contract { 44 } else { 12 })?,
                modular_additions: if contract { scaled(24)? } else { 0 },
                modular_multiplications: if contract { scaled(24)? } else { 0 },
                ..LogicalOperations::default()
            },
            coefficient_reductions: scaled(12)?,
        };
        for source in self.sources {
            work.add_ntt(source.operations(Operation::Inverse))?;
            for target in self.targets {
                work.add_ntt(target.operations(Operation::Forward))?;
            }
        }
        Ok(work)
    }

    fn preflight(
        &self,
        source: &SourceSpectra<'_>,
        rows: Option<&RowSpectra<'_>>,
        output: &[u64],
        scratch: &[u64],
        ceiling: Work,
    ) -> Result<Storage, Error> {
        let evaluation = if rows.is_some() { Evaluation::Contract } else { Evaluation::Embed };
        let storage = self.storage(evaluation)?;
        let work = self.work(evaluation)?;
        if source.contexts != self.sources
            || rows.is_some_and(|rows| rows.contexts != self.targets)
        {
            return Err(Error::ContextMismatch);
        }
        if source.values.len() != storage.source_words
            || rows.is_some_and(|rows| rows.values.len() != storage.row_words)
            || output.len() != storage.output_words
            || scratch.len() < storage.scratch_words
        {
            return Err(Error::BufferLength);
        }
        if !work.fits(ceiling) {
            return Err(Error::WorkLimit);
        }
        let n = self.degree();
        let mut invalid = false;
        for (i, context) in self.sources.iter().enumerate() {
            for &value in &source.values[i * n..(i + 1) * n] {
                invalid |= value >= context.ring().modulus();
            }
        }
        if let Some(rows) = rows {
            // Never stop early: include final row, component, limb, coefficient.
            for row in 0..3 {
                for component in 0..2 {
                    for (limb, context) in self.targets.iter().enumerate() {
                        let start = ((row * 2 + component) * 4 + limb) * n;
                        for &value in &rows.values[start..start + n] {
                            invalid |= value >= context.ring().modulus();
                        }
                    }
                }
            }
        }
        if invalid { Err(Error::NonCanonical) } else { Ok(storage) }
    }

    /// d_i is the canonical unsigned coefficient vector of inverse_NTT_qi(c_i).
    /// Output[i][limb] = NTT_target(d_i mod target), not centered or CRT lift.
    /// Exact input/output lengths; scratch >=2N, unused tail unchanged. Every
    /// Result refusal leaves ALL scratch/output words untouched (also tails).
    pub fn embed_into(
        &self,
        source: SourceSpectra<'_>,
        output: &mut [u64],
        scratch: &mut [u64],
        ceiling: Work,
    ) -> Result<(), Error> {
        self.preflight(&source, None, output, scratch, ceiling)?;
        let n = self.degree();
        let (coefficients, lifted) = scratch[..2 * n].split_at_mut(n);
        for i in 0..3 {
            self.sources[i].inverse_into(&source.values[i * n..(i + 1) * n], coefficients)
                .expect("preflight validated source spectrum and exact buffers");
            for (limb, target) in self.targets.iter().enumerate() {
                for (value, &digit) in lifted.iter_mut().zip(coefficients.iter()) {
                    *value = digit % target.ring().modulus();
                }
                let start = (i * 4 + limb) * n;
                target.forward_into(lifted, &mut output[start..start + n])
                    .expect("lift is canonical and buffers are exact");
            }
        }
        Ok(())
    }

    /// (v0,v1) = sum_i d_i*(b_i,a_i) in each target negacyclic ring; output
    /// remains natural-order NTT spectra. Source inverse is performed once per
    /// row, target forward once per row/limb, reused for both components.
    /// Accumulate modulo each target after EACH row. Every product <2^122;
    /// every canonical sum <2^62. No machine-word reconstruction of p*q or Q.
    /// Exact lengths; scratch >=3N, unused tail unchanged. Safe exclusive mutable
    /// borrows require scratch/output disjoint from each other and immutable
    /// inputs. All Result refusals precede ANY scratch/output write; validated
    /// primitive calls cannot subsequently refuse. No rollback allocation.
    pub fn contract_into(
        &self,
        source: SourceSpectra<'_>,
        rows: RowSpectra<'_>,
        output: &mut [u64],
        scratch: &mut [u64],
        ceiling: Work,
    ) -> Result<(), Error> {
        self.preflight(&source, Some(&rows), output, scratch, ceiling)?;
        self.contract_validated_into(source, rows, output, scratch);
        Ok(())
    }

    // Exact former contract_into execution body: no additional validation,
    // arithmetic changes, or fallible return after the caller's preflight.
    fn contract_validated_into(
        &self,
        source: SourceSpectra<'_>,
        rows: RowSpectra<'_>,
        output: &mut [u64],
        scratch: &mut [u64],
    ) {
        let n = self.degree();
        let (coefficients, rest) = scratch[..3 * n].split_at_mut(n);
        let (lifted, spectrum) = rest.split_at_mut(n);
        output.fill(0);
        for i in 0..3 {
            self.sources[i].inverse_into(&source.values[i * n..(i + 1) * n], coefficients)
                .expect("preflight validated source spectrum and exact buffers");
            for (limb, target) in self.targets.iter().enumerate() {
                let q = target.ring().modulus();
                for (value, &digit) in lifted.iter_mut().zip(coefficients.iter()) {
                    *value = digit % q;
                }
                target.forward_into(lifted, spectrum)
                    .expect("lift is canonical and buffers are exact");
                for component in 0..2 {
                    let row_start = ((i * 2 + component) * 4 + limb) * n;
                    let out_start = (component * 4 + limb) * n;
                    for k in 0..n {
                        let product = mul(spectrum[k], rows.values[row_start + k], q);
                        let sum = output[out_start + k] + product;
                        output[out_start + k] = if sum >= q { sum - q } else { sum };
                    }
                }
            }
        }
    }

}

#[cfg(test)]
#[path = "coefficient_rns_contraction/tests.rs"]
mod tests;
