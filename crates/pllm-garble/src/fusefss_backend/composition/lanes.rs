//! Repeat an immutable scalar/row template in its already-reserved public arrays.
//! Tables are retained once; each new instruction has a distinct PC/material.
use super::*;

impl Plan<'_> {
    pub fn repetition_bound(&self) -> usize {
        if self.is_atomic_vector() || self.inputs.is_empty() || self.code.is_empty() {
            return 1;
        }
        (MAX_INPUTS / self.inputs.len())
            .min(MAX_INSTRUCTIONS / self.code.len())
            .min(MAX_REGISTERS / self.registers.len())
            .min(MAX_REGISTERS / self.outputs.len())
            .min(
                MAX_WORK
                    / (self.resources.cumulative_work + 4 * self.resources.instructions).max(1),
            )
            .min(wavefront::MAX_ACTIVE)
    }

    /// Lane-major inputs and outputs, with the original per-lane domain. No
    /// register, instruction, table or work ceiling is raised. Consumes on error.
    pub fn repeat(mut self, lanes: usize) -> Result<Self, Error> {
        if lanes == 0 || lanes > self.repetition_bound() {
            return Err(Error::Resource);
        }
        if lanes == 1 {
            return Ok(self);
        }
        let inputs = self.inputs.len();
        let code = self.code.len();
        let outputs = self.outputs.len();
        let registers = self.registers.len();
        let map = |v: Value, lane: usize| {
            if v < inputs {
                lane * inputs + v
            } else {
                inputs * lanes + lane * code + v - inputs
            }
        };
        let remap = |i: Instruction, lane| {
            let v = |v| map(v, lane);
            match i {
                Instruction::Constant(n) => Instruction::Constant(n),
                Instruction::Add(a, b) => Instruction::Add(v(a), v(b)),
                Instruction::Sub(a, b) => Instruction::Sub(v(a), v(b)),
                Instruction::PublicMultiply(a, c) => Instruction::PublicMultiply(v(a), c),
                Instruction::Narrow(a) => Instruction::Narrow(v(a)),
                Instruction::Convert(a, op) => Instruction::Convert(v(a), op),
                Instruction::Multiply(a, b) => Instruction::Multiply(v(a), v(b)),
                Instruction::Predicate(a, s, t) => Instruction::Predicate(v(a), s, t),
                Instruction::Select(a, b, c) => Instruction::Select(v(a), v(b), v(c)),
                Instruction::Lookup(a, t) => Instruction::Lookup(v(a), t),
                Instruction::Gelu(a) => Instruction::Gelu(v(a)),
                Instruction::CompactTokenOneHot(_) => unreachable!("atomic plans cannot repeat"),
            }
        };
        if self.inputs.capacity() < inputs * lanes
            || self.code.capacity() < code * lanes
            || self.registers.capacity() < registers * lanes
            || self.outputs.capacity() < outputs * lanes
        {
            return Err(Error::Resource);
        }
        for lane in 1..lanes {
            for i in 0..inputs {
                self.inputs.push(self.inputs[i]);
            }
            for i in 0..code {
                self.code.push(remap(self.code[i], lane));
            }
            for i in 0..outputs {
                self.outputs.push(map(self.outputs[i], lane));
            }
        }
        for i in 0..code {
            self.code[i] = remap(self.code[i], 0);
        }
        for i in 0..outputs {
            self.outputs[i] = map(self.outputs[i], 0);
        }
        self.registers.resize(
            registers * lanes,
            Register {
                bits: 50,
                domain: Domain::Arithmetic,
            },
        );
        self.registers
            .copy_within(inputs..registers, inputs * lanes);
        for lane in 1..lanes {
            self.registers.copy_within(0..inputs, lane * inputs);
            self.registers.copy_within(
                inputs * lanes..inputs * lanes + code,
                inputs * lanes + lane * code,
            );
        }
        let r = &mut self.resources;
        for value in [
            &mut r.instructions,
            &mut r.registers,
            &mut r.secret_steps,
            &mut r.dcf_keys_per_party,
            &mut r.dcf_evaluations_per_party,
            &mut r.triples_per_party,
            &mut r.frames_per_party,
            &mut r.frame_bytes_per_party,
            &mut r.dcf_keys_total,
            &mut r.dcf_evaluations_total,
            &mut r.triples_total,
            &mut r.frames_total,
            &mut r.frame_bytes_total,
            &mut r.private_material_bytes_total,
            &mut r.public_table_copy_bytes_total,
            &mut r.cumulative_work,
        ] {
            *value = value.checked_mul(lanes).ok_or(Error::Resource)?;
        }
        r.cumulative_work = r
            .cumulative_work
            .checked_add(4 * r.instructions)
            .ok_or(Error::Resource)?;
        if r.cumulative_work > MAX_WORK {
            return Err(Error::Resource);
        }
        let owner = runtime::owner_bytes(self.registers.len(), self.outputs.len());
        r.combined_reservation_bytes = r
            .combined_reservation_bytes
            .checked_sub(2 * r.register_owner_bytes_per_party)
            .and_then(|v| v.checked_add(2 * owner))
            .ok_or(Error::Resource)?;
        r.register_owner_bytes_per_party = owner;
        let probe = self.lease.budget.reserve(
            r.combined_reservation_bytes
                .checked_sub(r.plan_reservation_bytes)
                .ok_or(Error::Resource)?,
        )?;
        drop(probe);
        self.digest = Sha256::new()
            .chain_update(b"pllm.fss.independent-template-lanes.v1\0")
            .chain_update(self.digest)
            .chain_update((lanes as u64).to_le_bytes())
            .finalize()
            .into();
        Ok(self)
    }
}
