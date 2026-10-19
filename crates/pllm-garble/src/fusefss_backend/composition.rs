//! Immutable, bounded scalar arithmetic schedules over the existing primitives.
//!
//! Public lowering contracts are commitments, not proofs of secret input ranges.
//! Runtime never issues preprocessing and owns exactly one party's registers.
//! Dealer material is explicitly issued by public instruction index, permitting
//! bounded streaming without an offline transport or SDK integration claim.
use super::{
    allocate, conversions, lookup, predicates, Budget, Error, Lease, MAX_ALLOCATION_BYTES,
};
use crate::{fusefss::Descriptor, shared_rescale_reference::Context};
use sha2::{Digest, Sha256};
use std::{fmt::Write, mem::size_of};

#[cfg(test)]
mod conformance;
pub mod gates;
pub mod norm;
mod runtime;
mod source_tables;
pub use runtime::{Event, Material, Runtime};

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
    /// Cumulative public payload copied into the two native lookup parties.
    /// Original immutable tables remain separate live plan owners.
    pub public_table_copy_bytes_total: usize,
    /// Conservative bit-level work: DCF evaluations times input width, plus
    /// local instructions and primitive triples/VM instructions.
    pub cumulative_work: usize,
    pub plan_reservation_bytes: usize,
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
    tables: Vec<lookup::Table<'a>>,
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
    tables: Vec<lookup::Table<'a>>,
    resources: Resources,
    lease: Lease<'a>,
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
        let bytes = self
            .resources
            .plan_reservation_bytes
            .checked_add(table.resources(1)?.table_reservation_bytes)
            .ok_or(Error::Resource)?;
        if bytes > MAX_PLAN_BYTES {
            return Err(Error::Resource);
        }
        let id = self.tables.len();
        self.hash.update(b"table\0");
        self.hash.update(table.digest());
        self.resources.plan_reservation_bytes = bytes;
        self.resources.public_table_reservation_bytes +=
            table.resources(1)?.table_reservation_bytes;
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
            resources: self.resources,
            lease: self.lease,
        })
    }
}

impl Plan<'_> {
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn resources(&self) -> Resources {
        self.resources
    }
    pub fn input_bits(&self) -> &[u8] {
        &self.inputs
    }
    pub fn output_bits(&self, index: usize) -> Result<u8, Error> {
        self.outputs
            .get(index)
            .map(|&v| self.registers[v].bits)
            .ok_or(Error::Domain)
    }
    /// True denotes a native XOR-shared output; all other outputs are additive.
    pub fn output_is_xor(&self, index: usize) -> Result<bool, Error> {
        self.outputs
            .get(index)
            .map(|&v| self.registers[v].domain == Domain::Xor)
            .ok_or(Error::Domain)
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
        let instruction = *self.code.get(pc).ok_or(Error::Domain)?;
        let output = self.registers[self.inputs.len() + pc].bits;
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
            input_bits: self.registers[value].bits,
            output_bits: output,
            lanes: 1,
            operation,
            source_digest: self.source,
            profile_digest: self.digest,
        })
    }
}
