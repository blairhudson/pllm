//! Immutable, bounded arithmetic schedules over the existing primitives.
//! Compact token one-hot is a standalone single-step plan, not a scalar register
//! expansion. Ordinary scalar instruction, register and input limits are unchanged.
//!
//! Public lowering contracts are commitments, not proofs of secret input ranges.
//! Runtime never issues preprocessing and owns exactly one party's registers.
//! Dealer material is explicitly issued by public instruction index, permitting
//! bounded streaming without an offline transport or SDK integration claim.
use super::{
    allocate, conversions, lookup, one_hot, predicates, Budget, Error, Lease, MAX_ALLOCATION_BYTES,
};
use crate::{fusefss::Descriptor, shared_rescale_reference::Context};
use sha2::{Digest, Sha256};
use std::{fmt::Write, mem::size_of};

pub mod cache;
#[cfg(test)]
mod conformance;
pub mod gates;
pub mod kernels;
mod lanes;
pub mod norm;
pub mod provider;
mod runtime;
mod source_tables;
pub mod wavefront;
pub use runtime::{wire, Engine, Event, Material, OwnedRuntime, Runtime};

#[cfg(test)]
#[path = "composition/compact_one_hot_tests.rs"]
mod compact_one_hot_tests;

pub(crate) type Value = usize;
pub const MAX_INSTRUCTIONS: usize = 32768;
pub const MAX_REGISTERS: usize = 65536;
pub const MAX_INPUTS: usize = 32768;
pub const MAX_TABLES: usize = 256;
pub const MAX_CONTRACT_BYTES: usize = 4096;
pub const MAX_PLAN_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_WORK: usize = 1 << 26;
const DESCRIPTOR_SCRATCH: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Domain {
    Arithmetic,
    Bit,
    Xor,
}
#[derive(Clone, Copy, Debug)]
struct Register {
    bits: u8,
    domain: Domain,
}
#[derive(Clone, Copy, Debug)]
enum Instruction {
    Constant(u64),
    Add(Value, Value),
    Sub(Value, Value),
    PublicMultiply(Value, u64),
    Narrow(Value),
    Convert(Value, conversions::Operation),
    Multiply(Value, Value),
    Predicate(Value, bool, i64),
    Select(Value, Value, Value),
    Lookup(Value, usize),
    Gelu(Value),
    // Whole-plan tag. Builder rejects this variant; its vector output belongs to
    // the native one-hot owner, never to the scalar register/output arrays.
    CompactTokenOneHot(usize),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Resources {
    pub instructions: usize,
    pub registers: usize,
    pub secret_steps: usize,
    pub dcf_keys_per_party: usize,
    pub dcf_evaluations_per_party: usize,
    pub triples_per_party: usize,
    pub frames_per_party: usize,
    pub frame_bytes_per_party: usize,
    pub dcf_keys_total: usize,
    pub dcf_evaluations_total: usize,
    /// Total native triple shares across both roles (two per paired product).
    pub triples_total: usize,
    pub frames_total: usize,
    pub frame_bytes_total: usize,
    /// Logical native private payload across both roles, not serialized wire.
    pub private_material_bytes_total: usize,
    /// Cumulative public ownership copies into both native lookup parties:
    /// payload for local tables, Arc handles for cached lowered chunks.
    /// Immutable backing remains a separate charged live owner.
    pub public_table_copy_bytes_total: usize,
    /// Conservative bit-level work: DCF evaluations times input width, plus
    /// local instructions and primitive triples/VM instructions. Compact one-hot
    /// additionally charges its full-domain PRG blocks and derivative row work.
    pub cumulative_work: usize,
    pub plan_reservation_bytes: usize,
    /// Optional compact-template construction peak, including old/new overlap.
    pub template_construction_bytes: usize,
    /// Public metadata actually copied when replacing the expanded arrays.
    pub template_copy_bytes: usize,
    /// Included in plan_reservation_bytes, held by independent Table leases.
    pub public_table_reservation_bytes: usize,
    pub register_owner_bytes_per_party: usize,
    pub largest_primitive_reservation_bytes: usize,
    /// Previous primitive's entire exported frame set, both roles. Caller must
    /// retire older sets; every actual issuance still atomically checks Budget.
    pub retained_frame_overlap_bytes: usize,
    pub combined_reservation_bytes: usize,
}

#[derive(Clone, Copy)]
struct NativeCost {
    peak: usize,
    keys: usize,
    evaluations: usize,
    triples: usize,
    frames: usize,
    frame_bytes: usize,
    private_bytes_total: usize,
    public_copy_bytes_total: usize,
    work: usize,
}
impl NativeCost {
    fn conversion(c: conversions::Resources, bits: u8) -> Self {
        Self {
            peak: c.combined_reservation_bytes,
            keys: c.dcf_keys_per_party,
            evaluations: c.dcf_evaluations_per_party,
            triples: c.triples,
            frames: 1,
            frame_bytes: c.frame_bytes_per_party,
            private_bytes_total: c.private_material_bytes_total,
            public_copy_bytes_total: 0,
            work: c.dcf_evaluations_per_party * bits as usize + c.triples,
        }
    }
}

pub struct Builder<'a> {
    source: [u8; 32],
    hash: Sha256,
    inputs: Vec<u8>,
    code: Vec<Instruction>,
    registers: Vec<Register>,
    outputs: Vec<Value>,
    tables: Vec<TableOwner<'a>>,
    resources: Resources,
    lease: Lease<'a>,
}
pub struct Plan<'a> {
    source: [u8; 32],
    digest: [u8; 32],
    inputs: Vec<u8>,
    code: Vec<Instruction>,
    registers: Vec<Register>,
    outputs: Vec<Value>,
    tables: Vec<TableOwner<'a>>,
    // A compact tensor retains one template, addressed by virtual lane/PC IDs.
    lanes: usize,
    resources: Resources,
    lease: Lease<'a>,
}

enum TableOwner<'a> {
    Local(lookup::Table<'a>),
    Shared(std::sync::Arc<cache::SharedTable<'a>>),
}
impl<'a> std::ops::Deref for TableOwner<'a> {
    type Target = lookup::Table<'a>;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Local(table) => table,
            Self::Shared(table) => &table.table,
        }
    }
}
impl TableOwner<'_> {
    fn reservation_bytes(&self) -> Result<usize, Error> {
        let table = self.resources(1)?.table_reservation_bytes;
        Ok(table
            + if matches!(self, Self::Shared(_)) {
                cache::shared_metadata_bytes()
            } else {
                0
            })
    }
}

fn width(bits: u8) -> Result<(), Error> {
    if (1..=50).contains(&bits) {
        Ok(())
    } else {
        Err(Error::Domain)
    }
}
fn mask(bits: u8) -> u64 {
    (1u64 << bits) - 1
}
fn descriptor_error(error: crate::fusefss::Error) -> Error {
    match error {
        crate::fusefss::Error::Allocation => Error::Allocation,
        _ => Error::Domain,
    }
}

// Debug is a versioned, typed encoding here, streamed directly into SHA-256;
// no formatting String or unbounded parser is involved. All enum fields are
// fixed-size scalar public metadata (including SignedRange's private fields).
struct HashWriter<'a>(&'a mut Sha256);
impl std::fmt::Write for HashWriter<'_> {
    fn write_str(&mut self, text: &str) -> std::fmt::Result {
        self.0.update(text.as_bytes());
        Ok(())
    }
}
fn typed_hash(hash: &mut Sha256, value: &impl std::fmt::Debug) {
    write!(HashWriter(hash), "{value:?}").expect("hash writer cannot fail");
}

impl<'a> Builder<'a> {
    pub(crate) fn new(
        source_digest: [u8; 32],
        contract: &[u8],
        input_bits: &[u8],
        budget: &'a Budget,
    ) -> Result<Self, Error> {
        if source_digest == [0; 32]
            || contract.is_empty()
            || contract.len() > MAX_CONTRACT_BYTES
            || input_bits.len() > MAX_INPUTS
        {
            return Err(Error::Domain);
        }
        for &bits in input_bits {
            width(bits)?;
        }
        let bytes = size_of::<Self>().max(size_of::<Plan<'_>>())
            + 4096
            + DESCRIPTOR_SCRATCH
            + MAX_INPUTS
            + MAX_INSTRUCTIONS * size_of::<Instruction>()
            + MAX_REGISTERS * (size_of::<Register>() + size_of::<Value>())
            + MAX_TABLES * size_of::<lookup::Table<'_>>();
        // Shared handles add one enum tag per slot. Existing 4096-byte owner
        // slack covers all tags; preserve default F2 reservation identities.
        if MAX_TABLES * (size_of::<TableOwner<'_>>() - size_of::<lookup::Table<'_>>()) > 4096 {
            return Err(Error::Resource);
        }
        if bytes > MAX_PLAN_BYTES {
            return Err(Error::Resource);
        }
        let lease = budget.reserve(bytes)?;
        let mut inputs = allocate(MAX_INPUTS, budget)?;
        inputs.extend_from_slice(input_bits);
        let mut registers = allocate(MAX_REGISTERS, budget)?;
        registers.extend(input_bits.iter().map(|&bits| Register {
            bits,
            domain: Domain::Arithmetic,
        }));
        let mut hash = Sha256::new();
        hash.update(b"pllm.fusefss.scalar-composition.typed-debug.v1\0");
        hash.update(source_digest);
        hash.update((contract.len() as u64).to_le_bytes());
        hash.update(contract);
        hash.update((input_bits.len() as u64).to_le_bytes());
        hash.update(input_bits);
        Ok(Self {
            source: source_digest,
            hash,
            inputs,
            registers,
            code: allocate(MAX_INSTRUCTIONS, budget)?,
            outputs: allocate(MAX_REGISTERS, budget)?,
            tables: allocate(MAX_TABLES, budget)?,
            resources: Resources {
                plan_reservation_bytes: bytes,
                ..Resources::default()
            },
            lease,
        })
    }
    pub(crate) fn input(&self, index: usize) -> Result<Value, Error> {
        if index < self.inputs.len() {
            Ok(index)
        } else {
            Err(Error::Domain)
        }
    }
    fn reg(&self, value: Value) -> Result<Register, Error> {
        self.registers.get(value).copied().ok_or(Error::Domain)
    }
    fn arithmetic(&self, value: Value) -> Result<Register, Error> {
        let r = self.reg(value)?;
        if r.domain == Domain::Xor {
            Err(Error::Domain)
        } else {
            Ok(r)
        }
    }
    fn pair(&self, left: Value, right: Value) -> Result<Register, Error> {
        let a = self.arithmetic(left)?;
        let b = self.arithmetic(right)?;
        if a.bits != b.bits {
            return Err(Error::Domain);
        }
        Ok(a)
    }
    fn room(&self, count: usize) -> Result<(), Error> {
        if self.code.len() + count > MAX_INSTRUCTIONS
            || self.registers.len() + count > MAX_REGISTERS
        {
            Err(Error::Resource)
        } else {
            Ok(())
        }
    }
    fn push(&mut self, instruction: Instruction, register: Register) -> Result<Value, Error> {
        if matches!(instruction, Instruction::CompactTokenOneHot(_)) {
            return Err(Error::Unsupported);
        }
        self.room(1)?;
        if self.resources.cumulative_work >= MAX_WORK {
            return Err(Error::Resource);
        }
        let id = self.registers.len();
        typed_hash(&mut self.hash, &instruction);
        typed_hash(&mut self.hash, &register);
        self.code.push(instruction);
        self.registers.push(register);
        self.resources.cumulative_work += 1;
        Ok(id)
    }
    fn charge(&mut self, cost: NativeCost) -> Result<(), Error> {
        let next = self
            .resources
            .cumulative_work
            .checked_add(cost.work)
            .ok_or(Error::Resource)?;
        if next >= MAX_WORK {
            return Err(Error::Resource);
        }
        fn add(a: usize, b: usize) -> Result<usize, Error> {
            a.checked_add(b).ok_or(Error::Resource)
        }
        let mut resources = self.resources;
        resources.cumulative_work = next;
        resources.secret_steps = add(resources.secret_steps, 1)?;
        resources.dcf_keys_per_party = add(resources.dcf_keys_per_party, cost.keys)?;
        resources.dcf_evaluations_per_party =
            add(resources.dcf_evaluations_per_party, cost.evaluations)?;
        resources.triples_per_party = add(resources.triples_per_party, cost.triples)?;
        resources.frames_per_party = add(resources.frames_per_party, cost.frames)?;
        resources.frame_bytes_per_party = add(resources.frame_bytes_per_party, cost.frame_bytes)?;
        resources.private_material_bytes_total = add(
            resources.private_material_bytes_total,
            cost.private_bytes_total,
        )?;
        resources.public_table_copy_bytes_total = add(
            resources.public_table_copy_bytes_total,
            cost.public_copy_bytes_total,
        )?;
        resources.dcf_keys_total = add(resources.dcf_keys_total, add(cost.keys, cost.keys)?)?;
        resources.dcf_evaluations_total = add(
            resources.dcf_evaluations_total,
            add(cost.evaluations, cost.evaluations)?,
        )?;
        resources.triples_total = add(resources.triples_total, add(cost.triples, cost.triples)?)?;
        resources.frames_total = add(resources.frames_total, add(cost.frames, cost.frames)?)?;
        let frames = add(cost.frame_bytes, cost.frame_bytes)?;
        resources.frame_bytes_total = add(resources.frame_bytes_total, frames)?;
        resources.largest_primitive_reservation_bytes =
            resources.largest_primitive_reservation_bytes.max(cost.peak);
        resources.retained_frame_overlap_bytes = resources.retained_frame_overlap_bytes.max(frames);
        self.resources = resources;
        Ok(())
    }
    fn conversion_cost(
        &self,
        input: u8,
        output: u8,
        operation: conversions::Operation,
    ) -> Result<conversions::Resources, Error> {
        conversions::resources(conversions::Spec {
            input_bits: input,
            output_bits: output,
            lanes: 1,
            operation,
            source_digest: self.source,
            profile_digest: [1; 32],
        })
    }
    pub(crate) fn constant(&mut self, bits: u8, value: u64) -> Result<Value, Error> {
        width(bits)?;
        if value > mask(bits) {
            return Err(Error::Domain);
        }
        self.push(
            Instruction::Constant(value),
            Register {
                bits,
                domain: if value <= 1 {
                    Domain::Bit
                } else {
                    Domain::Arithmetic
                },
            },
        )
    }
    pub(crate) fn add(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        let r = self.pair(left, right)?;
        self.push(
            Instruction::Add(left, right),
            Register {
                domain: Domain::Arithmetic,
                ..r
            },
        )
    }
    pub(crate) fn sub(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        let r = self.pair(left, right)?;
        self.push(
            Instruction::Sub(left, right),
            Register {
                domain: Domain::Arithmetic,
                ..r
            },
        )
    }
    pub(crate) fn mul_public(&mut self, value: Value, coefficient: u64) -> Result<Value, Error> {
        let r = self.arithmetic(value)?;
        self.push(
            Instruction::PublicMultiply(value, coefficient),
            Register {
                domain: Domain::Arithmetic,
                ..r
            },
        )
    }
    pub(crate) fn narrow(&mut self, value: Value, bits: u8) -> Result<Value, Error> {
        width(bits)?;
        let r = self.arithmetic(value)?;
        if bits > r.bits {
            return Err(Error::Domain);
        }
        self.push(Instruction::Narrow(value), Register { bits, ..r })
    }
    pub(crate) fn convert(
        &mut self,
        value: Value,
        operation: conversions::Operation,
        output_bits: u8,
    ) -> Result<Value, Error> {
        self.room(1)?;
        let r = self.reg(value)?;
        let b2a = matches!(operation, conversions::Operation::BooleanToArithmetic);
        if b2a != (r.domain == Domain::Xor)
            || matches!(
                operation,
                conversions::Operation::Multiply | conversions::Operation::PhiSelect { .. }
            )
        {
            return Err(Error::Domain);
        }
        let c = self.conversion_cost(r.bits, output_bits, operation)?;
        self.charge(NativeCost::conversion(c, r.bits))?;
        let domain = if b2a
            || (r.domain == Domain::Bit && matches!(operation, conversions::Operation::ZeroExtend))
        {
            Domain::Bit
        } else {
            Domain::Arithmetic
        };
        self.push(
            Instruction::Convert(value, operation),
            Register {
                bits: output_bits,
                domain,
            },
        )
    }
    pub(crate) fn multiply(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        self.room(1)?;
        let r = self.pair(left, right)?;
        let c = self.conversion_cost(r.bits, r.bits, conversions::Operation::Multiply)?;
        self.charge(NativeCost::conversion(c, r.bits))?;
        self.push(
            Instruction::Multiply(left, right),
            Register {
                domain: Domain::Arithmetic,
                ..r
            },
        )
    }
    fn compare(&mut self, value: Value, signed: bool, threshold: i64) -> Result<Value, Error> {
        self.room(2)?;
        let r = self.arithmetic(value)?;
        if if signed {
            threshold < -(1i64 << (r.bits - 1)) || threshold >= 1i64 << (r.bits - 1)
        } else {
            threshold < 0 || threshold as u64 > 1u64 << r.bits
        } {
            return Err(Error::Domain);
        }
        self.lease.budget.allocation()?;
        let descriptor =
            Descriptor::simple_comparison(r.bits, signed, threshold).map_err(descriptor_error)?;
        let c = predicates::resources(&descriptor)?;
        let b = self.conversion_cost(1, r.bits, conversions::Operation::BooleanToArithmetic)?;
        let work = c.dcf_evaluations_per_party * r.bits as usize
            + descriptor.shape().vm_ops
            + c.triples_per_party
            + b.triples;
        if self
            .resources
            .cumulative_work
            .checked_add(work + 2)
            .ok_or(Error::Resource)?
            >= MAX_WORK
        {
            return Err(Error::Resource);
        }
        self.charge(NativeCost {
            peak: c.combined_reservation_bytes,
            keys: c.dcf_keys_per_party,
            evaluations: c.dcf_evaluations_per_party,
            triples: c.triples_per_party,
            frames: c.frames_per_party,
            frame_bytes: c.total_peer_frame_bytes / 2,
            // Exact typed simple-comparison program contains only Query,
            // Bit, Xor and And. Native Bit immediates are private u64
            // shares; query arguments/opcodes are public. Include native
            // input-mask share and triple shares, not just encoded DCFs.
            private_bytes_total: {
                let shape = descriptor.shape();
                let bits = shape
                    .vm_ops
                    .checked_sub(shape.queries + shape.boolean_xors + shape.boolean_ands)
                    .ok_or(Error::Domain)?;
                let key = crate::compact_dcf::RingComparison::resources(r.bits, 1)
                    .map_err(super::primitive_error)?;
                2 * (c.dcf_keys_per_party * key.encoded_key_bytes
                    + 8
                    + bits * 8
                    + c.triples_per_party * crate::shared_wide_product::ring::TRIPLE_BYTES)
            },
            work: work - b.triples,
            public_copy_bytes_total: 0,
        })?;
        let xor = self.push(
            Instruction::Predicate(value, signed, threshold),
            Register {
                bits: 1,
                domain: Domain::Xor,
            },
        )?;
        self.convert(xor, conversions::Operation::BooleanToArithmetic, r.bits)
    }
    pub(crate) fn lt_signed(&mut self, value: Value, threshold: i64) -> Result<Value, Error> {
        self.compare(value, true, threshold)
    }
    pub(crate) fn lt_unsigned(&mut self, value: Value, threshold: u64) -> Result<Value, Error> {
        if threshold > i64::MAX as u64 {
            return Err(Error::Domain);
        }
        self.compare(value, false, threshold as i64)
    }
    pub(crate) fn select(&mut self, bit: Value, yes: Value, no: Value) -> Result<Value, Error> {
        self.room(1)?;
        let r = self.pair(yes, no)?;
        let selector = self.reg(bit)?;
        if selector.domain != Domain::Bit || selector.bits != r.bits {
            return Err(Error::Domain);
        }
        let c = self.conversion_cost(
            r.bits,
            r.bits,
            conversions::Operation::PhiSelect {
                add: 0,
                bit_contract: [1; 32],
            },
        )?;
        self.charge(NativeCost::conversion(c, r.bits))?;
        let domain = if self.reg(yes)?.domain == Domain::Bit && self.reg(no)?.domain == Domain::Bit
        {
            Domain::Bit
        } else {
            Domain::Arithmetic
        };
        self.push(Instruction::Select(bit, yes, no), Register { domain, ..r })
    }
    pub(crate) fn table(&mut self, bits: u8, values: &[u64]) -> Result<usize, Error> {
        if self.tables.len() == MAX_TABLES {
            return Err(Error::Resource);
        }
        let table = lookup::Table::new_zero_extended(bits, values, self.source, self.lease.budget)?;
        self.retain_table(TableOwner::Local(table))
    }
    fn source_table(
        &mut self,
        cache: &cache::SourceTableCache<'a>,
        kind: cache::Kind,
    ) -> Result<usize, Error> {
        if self.tables.len() == MAX_TABLES {
            return Err(Error::Resource);
        }
        if !std::ptr::eq(self.lease.budget, cache.budget()) {
            return Err(Error::Domain);
        }
        let table = cache.get(kind, self.source)?;
        self.retain_table(TableOwner::Shared(table))
    }
    fn retain_table(&mut self, table: TableOwner<'a>) -> Result<usize, Error> {
        let reservation = table.reservation_bytes()?;
        let bytes = self
            .resources
            .plan_reservation_bytes
            .checked_add(reservation)
            .ok_or(Error::Resource)?;
        if bytes > MAX_PLAN_BYTES {
            return Err(Error::Resource);
        }
        let id = self.tables.len();
        self.hash.update(b"table\0");
        self.hash.update(table.digest());
        self.resources.plan_reservation_bytes = bytes;
        self.resources.public_table_reservation_bytes += reservation;
        self.tables.push(table);
        Ok(id)
    }
    pub(crate) fn lookup(&mut self, value: Value, table_index: usize) -> Result<Value, Error> {
        self.room(1)?;
        let r = self.arithmetic(value)?;
        let table = self.tables.get(table_index).ok_or(Error::Domain)?;
        if table.bits() != r.bits {
            return Err(Error::Domain);
        }
        let c = table.resources(1)?;
        self.charge(NativeCost {
            peak: c.issuance_reservation_bytes,
            keys: c.dcf_keys_per_party,
            evaluations: c.dcf_evaluations_per_party,
            triples: 0,
            frames: 1,
            frame_bytes: c.peer_frame_bytes_per_party,
            private_bytes_total: 2 * c.private_key_bytes_per_party,
            public_copy_bytes_total: c
                .public_table_bytes_per_party
                .checked_mul(2)
                .ok_or(Error::Resource)?,
            work: c.dcf_evaluations_per_party * r.bits as usize,
        })?;
        self.push(
            Instruction::Lookup(value, table_index),
            Register {
                domain: Domain::Arithmetic,
                ..r
            },
        )
    }
    pub(crate) fn gelu(&mut self, value: Value) -> Result<Value, Error> {
        self.room(1)?;
        let register = self.arithmetic(value)?;
        if register.bits != 50 {
            return Err(Error::Domain);
        }
        let gate = super::CompiledGelu::with_backend(
            self.lease.budget,
            super::LookupBackend::PublicBoundary,
        )?;
        let cost = gate.resources(1)?;
        self.hash.update(b"source-gelu-public-boundary\0");
        self.hash.update(gate.descriptor_digest());
        self.charge(NativeCost {
            peak: cost.combined_reservation_bytes,
            keys: cost.issuance.dcf_keys_per_party,
            evaluations: cost.issuance.dcf_evaluations_per_party,
            triples: 1,
            frames: 2,
            frame_bytes: cost.issuance.total_peer_frame_bytes / 2,
            private_bytes_total: 2 * cost.issuance.party_key_logical_bytes,
            public_copy_bytes_total: 2 * cost.issuance.public_table_bytes_per_party,
            work: cost.issuance.dcf_evaluations_per_party * 50 + 1,
        })?;
        self.push(Instruction::Gelu(value), register)
    }
    pub(crate) fn finish(mut self, outputs: &[Value]) -> Result<Plan<'a>, Error> {
        if outputs.is_empty() || outputs.len() > MAX_REGISTERS {
            return Err(Error::Domain);
        }
        for &v in outputs {
            self.reg(v)?;
        }
        self.outputs.extend_from_slice(outputs);
        self.hash.update(b"outputs\0");
        self.hash.update((outputs.len() as u64).to_le_bytes());
        for &v in outputs {
            self.hash.update((v as u64).to_le_bytes());
        }
        self.resources.instructions = self.code.len();
        self.resources.registers = self.registers.len();
        let owner = runtime::owner_bytes(self.registers.len(), outputs.len());
        self.resources.register_owner_bytes_per_party = owner;
        let combined = self
            .resources
            .plan_reservation_bytes
            .checked_add(2 * owner)
            .and_then(|v| v.checked_add(self.resources.largest_primitive_reservation_bytes))
            .and_then(|v| v.checked_add(self.resources.retained_frame_overlap_bytes))
            .ok_or(Error::Resource)?;
        if combined > MAX_ALLOCATION_BYTES {
            return Err(Error::Resource);
        }
        self.resources.combined_reservation_bytes = combined;
        // Atomic public preflight against actual shared Budget headroom, not
        // merely its configured limit. The immutable Plan/Table owners are
        // already charged. This probe is not a concurrency guarantee: online
        // register construction and each native issuance reserve again.
        let probe = self.lease.budget.reserve(
            combined
                .checked_sub(self.resources.plan_reservation_bytes)
                .ok_or(Error::Resource)?,
        )?;
        drop(probe);
        Ok(Plan {
            source: self.source,
            digest: self.hash.finalize().into(),
            inputs: self.inputs,
            code: self.code,
            registers: self.registers,
            outputs: self.outputs,
            tables: self.tables,
            lanes: 0,
            resources: self.resources,
            lease: self.lease,
        })
    }
}

impl<'a> Plan<'a> {
    /// Called only by the declared-range-validating kernel entrypoint. The native
    /// primitive owns the complete output and its lease; no per-vocabulary Plan
    /// metadata or second output vector is allocated.
    fn compact_token_one_hot(
        source: [u8; 32],
        contract: [u8; 32],
        vocabulary: usize,
        budget: &'a Budget,
    ) -> Result<Self, Error> {
        let spec = one_hot::Spec {
            vocabulary,
            source_digest: source,
            profile_digest: contract,
        };
        let native = one_hot::resources(spec)?;
        // Commit the existing backend identity as well as the kernel's source,
        // declared membership interval and parent profile. Issuance below uses
        // this complete Plan digest as its own profile binding.
        let digest = Sha256::new()
            .chain_update(b"pllm.fusefss.composition.compact-token-one-hot.v1\0")
            .chain_update(spec.digest()?)
            .finalize()
            .into();
        let plan_bytes = size_of::<Self>()
            + 4096 // Arc counters, hashing and fixed owner/alignment scratch.
            + size_of::<u8>()
            + size_of::<Instruction>()
            + size_of::<Register>();
        let owner = runtime::owner_bytes(1, 0);
        let work = native
            .dcf_evaluations_per_party
            .checked_mul(usize::from(native.input_bits))
            .and_then(|n| n.checked_add(native.prg_blocks_per_party))
            .and_then(|n| n.checked_add(8 * native.derivative_rows_per_party))
            .and_then(|n| n.checked_add(1))
            .ok_or(Error::Resource)?;
        let combined = plan_bytes
            .checked_add(2 * owner)
            .and_then(|n| n.checked_add(native.combined_reservation_bytes))
            .and_then(|n| n.checked_add(native.frame_bytes_total))
            .ok_or(Error::Resource)?;
        if plan_bytes > MAX_PLAN_BYTES || work > MAX_WORK || combined > MAX_ALLOCATION_BYTES {
            return Err(Error::Resource);
        }
        // Preflight the entire live envelope before even allocating metadata.
        // Online constructors and native issuance independently reserve again.
        let mut reservation = budget.reserve(combined)?;
        let mut inputs = allocate(1, budget)?;
        inputs.push(50);
        let mut code = allocate(1, budget)?;
        code.push(Instruction::CompactTokenOneHot(vocabulary));
        let mut registers = allocate(1, budget)?;
        registers.push(Register {
            bits: 50,
            domain: Domain::Arithmetic,
        });
        Ok(Self {
            source,
            digest,
            inputs,
            code,
            registers,
            outputs: Vec::new(),
            tables: Vec::new(),
            lanes: 0,
            resources: Resources {
                template_construction_bytes: 0,
                template_copy_bytes: 0,
                instructions: 1,
                registers: 1,
                secret_steps: 1,
                dcf_keys_per_party: native.dcf_keys_per_party,
                dcf_evaluations_per_party: native.dcf_evaluations_per_party,
                triples_per_party: native.triples_per_party,
                frames_per_party: native.frames_per_party,
                frame_bytes_per_party: native.frame_bytes_per_party,
                dcf_keys_total: 2 * native.dcf_keys_per_party,
                dcf_evaluations_total: 2 * native.dcf_evaluations_per_party,
                triples_total: 2 * native.triples_per_party,
                frames_total: 2 * native.frames_per_party,
                frame_bytes_total: native.frame_bytes_total,
                private_material_bytes_total: native.private_material_bytes_total,
                public_table_copy_bytes_total: 0,
                cumulative_work: work,
                plan_reservation_bytes: plan_bytes,
                public_table_reservation_bytes: 0,
                register_owner_bytes_per_party: owner,
                largest_primitive_reservation_bytes: native.combined_reservation_bytes,
                retained_frame_overlap_bytes: native.frame_bytes_total,
                combined_reservation_bytes: combined,
            },
            lease: reservation.split(plan_bytes)?,
        })
    }

    fn compact_vocabulary(&self) -> Option<usize> {
        match self.code.as_slice() {
            [Instruction::CompactTokenOneHot(vocabulary)] => Some(*vocabulary),
            _ => None,
        }
    }

    fn one_hot_spec(&self, pc: usize) -> Result<one_hot::Spec, Error> {
        if pc != 0 {
            return Err(Error::Domain);
        }
        Ok(one_hot::Spec {
            vocabulary: self.compact_vocabulary().ok_or(Error::Domain)?,
            source_digest: self.source,
            profile_digest: self.digest,
        })
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn resources(&self) -> Resources {
        self.resources
    }
    pub fn input_bits(&self) -> &[u8] {
        &self.inputs
    }
    /// Whole-vector lookup is already one atomic exchange, not a scalar DAG.
    pub fn is_atomic_vector(&self) -> bool {
        self.compact_vocabulary().is_some()
    }
    /// Exact output count. Only the explicit compact one-hot plan can exceed the
    /// ordinary scalar register cap; its native output is independently priced.
    pub fn output_count(&self) -> usize {
        self.compact_vocabulary()
            .unwrap_or(self.outputs.len() * self.lanes.max(1))
    }
    pub fn output_bits(&self, index: usize) -> Result<u8, Error> {
        if let Some(vocabulary) = self.compact_vocabulary() {
            return (index < vocabulary).then_some(50).ok_or(Error::Domain);
        }
        Ok(self.register(self.output_index(index)?)?.bits)
    }
    /// True denotes a native XOR-shared output; all other outputs are additive.
    pub fn output_is_xor(&self, index: usize) -> Result<bool, Error> {
        if let Some(vocabulary) = self.compact_vocabulary() {
            return (index < vocabulary).then_some(false).ok_or(Error::Domain);
        }
        Ok(self.register(self.output_index(index)?)?.domain == Domain::Xor)
    }
    fn step_context(&self, context: Context, pc: usize) -> Context {
        let operation = Sha256::new()
            .chain_update(b"pllm.fusefss.composition.step.v1\0")
            .chain_update(self.digest)
            .chain_update(context.plan)
            .chain_update(context.session)
            .chain_update(context.operation)
            .chain_update(context.first_tensor_index.to_le_bytes())
            .chain_update((pc as u64).to_le_bytes())
            .finalize()
            .into();
        Context {
            plan: self.digest,
            operation,
            ..context
        }
    }
    fn conversion_spec(&self, pc: usize) -> Result<conversions::Spec, Error> {
        let instruction = self.instruction(pc)?;
        let output = self.register(self.inputs.len() + pc)?.bits;
        let (value, operation) = match instruction {
            Instruction::Convert(v, op) => (v, op),
            Instruction::Multiply(v, _) => (v, conversions::Operation::Multiply),
            Instruction::Select(v, _, _) => (
                v,
                conversions::Operation::PhiSelect {
                    add: 0,
                    bit_contract: self.digest,
                },
            ),
            _ => return Err(Error::Domain),
        };
        Ok(conversions::Spec {
            input_bits: self.register(value)?.bits,
            output_bits: output,
            lanes: 1,
            operation,
            source_digest: self.source,
            profile_digest: self.digest,
        })
    }
}
