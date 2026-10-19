use super::*;
use zeroize::{Zeroize, Zeroizing};

/// Typed comparison query. Clear thresholds are dealer secrets, never evaluator
/// metadata. Getters exist solely for the explicit trusted backend handoff.
pub struct PredicateQuery {
    bits: u8,
    shift: u64,
    threshold: u64,
}
impl PredicateQuery {
    pub fn bits(&self) -> u8 {
        self.bits
    }
    pub fn shift(&self) -> u64 {
        self.shift
    }
    pub fn threshold(&self) -> u64 {
        self.threshold
    }
}
impl Zeroize for PredicateQuery {
    fn zeroize(&mut self) {
        self.bits.zeroize();
        self.shift.zeroize();
        self.threshold.zeroize();
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(u8)]
pub enum Opcode {
    Bit = 0,
    Query = 1,
    Not = 2,
    Xor = 3,
    And = 4,
    Or = 5,
}
/// Postfix Boolean VM instruction, not an arithmetic operation. Bit operands
/// include secret mask-dependent carries and must be shared by the backend.
pub struct Instruction {
    opcode: Opcode,
    argument: u32,
}
impl Instruction {
    pub fn opcode(&self) -> Opcode {
        self.opcode
    }
    pub fn argument(&self) -> u32 {
        self.argument
    }
}
impl Zeroize for Instruction {
    fn zeroize(&mut self) {
        self.opcode = Opcode::Bit;
        self.argument.zeroize();
    }
}

/// Mask-dependent clear plan, for trusted preprocessing only. Deliberately no
/// Debug, Clone, serialization, or evaluator API. All private vector contents and
/// retained private material is zeroized on drop, including partial construction
/// errors. The mask and source ordinals are retained only in test builds.
pub struct DealerPlan {
    digest: [u8; 32],
    word: ArithmeticType,
    shape: Shape,
    #[cfg(test)]
    mask: Zeroizing<u64>,
    boundaries: Zeroizing<Vec<u64>>,
    #[cfg(test)]
    ordinals: Zeroizing<Vec<u64>>,
    coefficients: Zeroizing<Vec<[u64; 2]>>,
    queries: Zeroizing<Vec<PredicateQuery>>,
    instructions: Zeroizing<Vec<Instruction>>,
}

/// Borrowed **trusted dealer backend** input. Never send this structure, its
/// clear boundaries, coefficient selections, carries, or query thresholds to an
/// evaluator. The backend must issue separate protected party keys and shares.
/// Horner consumes the original shared input, not the opened masked input; the
/// linear coefficient is Q0 and no extra Q12 rescale is implied.
pub struct DealerHandoff<'a> {
    pub descriptor_digest: [u8; 32],
    pub word: ArithmeticType,
    pub shape: Shape,
    pub shifted_boundaries: &'a [u64],
    /// Exactly [constant, linear] for every padded half-open interval.
    pub coefficients: &'a [[u64; 2]],
    pub predicate_queries: &'a [PredicateQuery],
    /// Concatenated postfix programs, delimited by shape.program_lengths.
    pub boolean_program: &'a [Instruction],
}
impl DealerPlan {
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    /// Invoke only inside trusted dealer issuance. The borrowed data cannot
    /// outlive this plan. A backend that copies any clear material owns erasure
    /// and admission of its copies; this interface does not authenticate roles.
    pub fn with_dealer_handoff<T>(
        &self,
        issue: impl FnOnce(DealerHandoff<'_>) -> Result<T>,
    ) -> Result<T> {
        issue(DealerHandoff {
            descriptor_digest: self.digest,
            word: self.word,
            shape: self.shape,
            shifted_boundaries: &self.boundaries,
            coefficients: &self.coefficients,
            predicate_queries: &self.queries,
            boolean_program: &self.instructions,
        })
    }
}

struct Emitter {
    bits: u8,
    mask: Zeroizing<u64>,
    keep: bool,
    queries: Zeroizing<Vec<PredicateQuery>>,
    instructions: Zeroizing<Vec<Instruction>>,
    query_count: usize,
    ops: usize,
    ands: usize,
    xors: usize,
    program_lengths: [usize; MAX_BOOLEAN_OUTPUTS],
}
impl Emitter {
    fn new(bits: u8, mask: u64, shape: Option<Shape>) -> Result<Self> {
        let mut queries = Zeroizing::new(Vec::new());
        let mut instructions = Zeroizing::new(Vec::new());
        if let Some(s) = shape {
            reserve(&mut queries, s.queries)?;
            reserve(&mut instructions, s.vm_ops)?;
        }
        Ok(Self {
            bits,
            mask: Zeroizing::new(mask),
            keep: shape.is_some(),
            queries,
            instructions,
            query_count: 0,
            ops: 0,
            ands: 0,
            xors: 0,
            program_lengths: [0; MAX_BOOLEAN_OUTPUTS],
        })
    }
    fn emit(&mut self, opcode: Opcode, argument: u32) -> Result<()> {
        self.ops = self.ops.checked_add(1).ok_or(Error::WorkBudget)?;
        if self.ops > MAX_VM_OPS {
            return Err(Error::WorkBudget);
        }
        self.ands += usize::from(matches!(opcode, Opcode::And | Opcode::Or));
        self.xors += if opcode == Opcode::Xor {
            1
        } else if opcode == Opcode::Or {
            2
        } else {
            0
        };
        if self.keep {
            if self.instructions.len() == self.instructions.capacity() {
                return Err(Error::Shape);
            }
            self.instructions.push(Instruction { opcode, argument });
        }
        Ok(())
    }
    fn query(&mut self, bits: u8, shift: u64, threshold: u64) -> Result<()> {
        let index = self.query_count;
        self.query_count = self.query_count.checked_add(1).ok_or(Error::WorkBudget)?;
        if self.query_count > MAX_QUERIES {
            return Err(Error::WorkBudget);
        }
        if self.keep {
            if self.queries.len() == self.queries.capacity() {
                return Err(Error::Shape);
            }
            self.queries.push(PredicateQuery {
                bits,
                shift,
                threshold,
            });
        }
        self.emit(Opcode::Query, index as u32)
    }
    fn lt(&mut self, bits: u8, shift: u64, threshold: u64) -> Result<()> {
        let n = 1u64 << bits;
        if threshold == 0 || threshold == n {
            return self.emit(Opcode::Bit, u32::from(threshold == n));
        }
        let r = *self.mask & (n - 1);
        self.query(bits, shift, (r + threshold) & (n - 1))?;
        self.query(bits, shift, r)?; // Keep both slots even for sampled zero thresholds.
        self.emit(Opcode::Xor, 0)?;
        self.emit(Opcode::Bit, u32::from(r + threshold >= n))?;
        self.emit(Opcode::Xor, 0)
    }
    fn expression(&mut self, nodes: &[Expr], index: usize) -> Result<()> {
        match nodes[index] {
            Expr::Const(v) => self.emit(Opcode::Bit, u32::from(v)),
            Expr::Lt(t) => self.lt(self.bits, 0, t),
            Expr::LowLt(k, t) => self.lt(k, 0, t),
            Expr::SignedLt(t) => {
                let half = 1u64 << (self.bits - 1);
                self.lt(self.bits, half, (t + half as i64) as u64)
            }
            Expr::Msb(c) => {
                self.lt(self.bits, c, 1u64 << (self.bits - 1))?;
                self.emit(Opcode::Not, 0)
            }
            Expr::Not(a) => {
                self.expression(nodes, a)?;
                self.emit(Opcode::Not, 0)
            }
            Expr::Xor(a, b) | Expr::And(a, b) | Expr::Or(a, b) => {
                self.expression(nodes, a)?;
                self.expression(nodes, b)?;
                self.emit(
                    match nodes[index] {
                        Expr::Xor(..) => Opcode::Xor,
                        Expr::And(..) => Opcode::And,
                        _ => Opcode::Or,
                    },
                    0,
                )
            }
        }
    }
    fn normalize(&mut self, d: &Descriptor) -> Result<()> {
        for output in 0..d.shape.boolean_outputs {
            let start = self.ops;
            for (i, p) in d.pieces.iter().enumerate() {
                self.lt(self.bits, 0, d.boundaries[i + 1])?;
                self.lt(self.bits, 0, d.boundaries[i])?;
                self.emit(Opcode::Xor, 0)?;
                self.expression(&d.expressions, p.roots[output])?;
                self.emit(Opcode::And, 0)?;
                if i != 0 {
                    self.emit(Opcode::Xor, 0)?;
                }
            }
            self.program_lengths[output] = self.ops - start;
        }
        Ok(())
    }
}

pub(super) fn admission(d: &Descriptor) -> Result<Shape> {
    let mut dry = Emitter::new(d.word.bits, 0, None)?;
    dry.normalize(d)?;
    let source = d.pieces.len();
    let padded = ((source + 1) as u64).min(1u64 << d.word.bits) as usize;
    let w = usize::from(d.word.bits).div_ceil(8);
    let b = (usize::from(d.word.bits) + 1).div_ceil(8);
    // All operands already have small public caps; use checked accumulation too.
    let terms = [
        56,
        w,
        (padded + 1) * b,
        padded * (2 + 2 * w),
        dry.query_count * (1 + 2 * w),
        d.shape.boolean_outputs * 4,
        dry.ops * 5,
    ];
    let size = terms
        .into_iter()
        .try_fold(0usize, |s, t| s.checked_add(t).ok_or(Error::InstanceBytes))?;
    if size > MAX_INSTANCE_BYTES {
        return Err(Error::InstanceBytes);
    }
    Ok(Shape {
        source_intervals: source,
        padded_intervals: padded,
        boolean_outputs: d.shape.boolean_outputs,
        queries: dry.query_count,
        vm_ops: dry.ops,
        boolean_ands: dry.ands,
        boolean_xors: dry.xors,
        reference_bytes: size,
        program_lengths: dry.program_lengths,
    })
}

pub(super) fn compile(d: &Descriptor, mask: u64) -> Result<DealerPlan> {
    let n = 1u64 << d.word.bits;
    if mask >= n {
        return Err(Error::Range);
    }
    let s = d.shape;
    // Fixed capacities determined entirely before reading any mask-dependent data.
    let mut starts = Zeroizing::new(Vec::<[u64; 2]>::new());
    reserve(&mut starts, s.source_intervals)?;
    let mut boundaries = Zeroizing::new(Vec::new());
    reserve(&mut boundaries, s.padded_intervals + 1)?;
    let mut ordinals = Zeroizing::new(Vec::new());
    reserve(&mut ordinals, s.padded_intervals)?;
    let mut coefficients = Zeroizing::new(Vec::new());
    reserve(&mut coefficients, s.padded_intervals)?;
    let mut emitter = Emitter::new(d.word.bits, mask, Some(s))?;
    for (i, &a) in d.boundaries[..s.source_intervals].iter().enumerate() {
        starts.push([(a + mask) & (n - 1), i as u64]);
    }
    starts.sort_unstable(); // In-place sort: dealer-only, no secret evaluator branches.
    if starts[0][0] != 0 {
        boundaries.push(0);
        ordinals.push(starts.last().ok_or(Error::Shape)?[1]);
        for &[a, i] in starts.iter() {
            boundaries.push(a);
            ordinals.push(i);
        }
        boundaries.push(n);
    } else {
        // First eligible split, including a possible final interval ending at N.
        let mut split = false;
        for (j, &[a, i]) in starts.iter().enumerate() {
            boundaries.push(a);
            ordinals.push(i);
            let end = starts.get(j + 1).map_or(n, |v| v[0]);
            if !split && (s.source_intervals as u64) < n && end - a >= 2 {
                boundaries.push(a + 1);
                ordinals.push(i);
                split = true;
            }
        }
        boundaries.push(n);
    }
    for &i in ordinals.iter() {
        coefficients.push(d.pieces[i as usize].coefficients);
    }
    emitter.normalize(d)?;
    if ordinals.len() != s.padded_intervals
        || boundaries.len() != s.padded_intervals + 1
        || emitter.query_count != s.queries
        || emitter.ops != s.vm_ops
        || emitter.program_lengths != s.program_lengths
    {
        return Err(Error::Shape);
    }
    Ok(DealerPlan {
        digest: d.digest,
        word: d.word,
        shape: s,
        #[cfg(test)]
        mask: Zeroizing::new(mask),
        boundaries,
        #[cfg(test)]
        ordinals,
        coefficients,
        queries: emitter.queries,
        instructions: emitter.instructions,
    })
}

#[cfg(test)]
#[test]
fn work_counters_fail_closed_at_caps() {
    let mut emitter = Emitter::new(50, 0, None).unwrap();
    for _ in 0..MAX_VM_OPS {
        emitter.emit(Opcode::Bit, 0).unwrap();
    }
    assert!(matches!(
        emitter.emit(Opcode::Bit, 0),
        Err(Error::WorkBudget)
    ));
    let mut emitter = Emitter::new(50, 0, None).unwrap();
    for _ in 0..MAX_QUERIES {
        emitter.query(50, 0, 0).unwrap();
    }
    assert!(matches!(emitter.query(50, 0, 0), Err(Error::WorkBudget)));
}

#[cfg(test)]
impl DealerPlan {
    pub(super) fn test_owner_capacities(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.boundaries.capacity() * std::mem::size_of::<u64>(),
            self.ordinals.capacity() * std::mem::size_of::<u64>(),
            self.coefficients.capacity() * std::mem::size_of::<[u64; 2]>(),
            self.queries.capacity() * std::mem::size_of::<PredicateQuery>(),
            self.instructions.capacity() * std::mem::size_of::<Instruction>(),
        )
    }
    pub(super) fn clear_interpret(
        &self,
        masked: u64,
        expected: [u8; 32],
    ) -> Result<(u64, Vec<bool>)> {
        if expected != self.digest {
            return Err(Error::DescriptorIdentity);
        }
        let n = 1u64 << self.word.bits;
        if masked >= n {
            return Err(Error::Range);
        }
        let i = self.boundaries.partition_point(|&v| v <= masked) - 1;
        let [c, l] = self.coefficients[i];
        let x = masked.wrapping_sub(*self.mask) & (n - 1);
        let result = ((u128::from(l) * u128::from(x) + u128::from(c)) & u128::from(n - 1)) as u64;
        let mut query_bits = Zeroizing::new(Vec::new());
        reserve(&mut query_bits, self.queries.len())?;
        for q in self.queries.iter() {
            query_bits.push(((masked + q.shift) & ((1u64 << q.bits) - 1)) < q.threshold);
        }
        let mut stack = Zeroizing::new(Vec::<bool>::new());
        reserve(&mut stack, self.shape.vm_ops)?;
        let mut outputs = Vec::new();
        reserve(&mut outputs, self.shape.boolean_outputs)?;
        let mut offset = 0;
        for &length in &self.shape.program_lengths[..self.shape.boolean_outputs] {
            stack.clear();
            for ins in &self.instructions[offset..offset + length] {
                match ins.opcode {
                    Opcode::Bit => stack.push(ins.argument != 0),
                    Opcode::Query => stack.push(query_bits[ins.argument as usize]),
                    Opcode::Not => {
                        let v = stack.last_mut().ok_or(Error::Type)?;
                        *v = !*v;
                    }
                    op => {
                        let r = stack.pop().ok_or(Error::Type)?;
                        let l = stack.pop().ok_or(Error::Type)?;
                        stack.push(match op {
                            Opcode::Xor => l ^ r,
                            Opcode::And => l & r,
                            Opcode::Or => l | r,
                            _ => return Err(Error::Type),
                        });
                    }
                }
            }
            if stack.len() != 1 {
                return Err(Error::Type);
            }
            outputs.push(stack[0]);
            offset += length;
        }
        Ok((result, outputs))
    }
    pub(super) fn reference_record(&self) -> Result<Zeroizing<Vec<u8>>> {
        let mut bytes = Zeroizing::new(Vec::new());
        reserve(&mut bytes, self.shape.reference_bytes)?;
        let n = self.word.bits;
        let w = usize::from(n).div_ceil(8);
        let b = (usize::from(n) + 1).div_ceil(8);
        bytes.extend_from_slice(b"FFSREF01");
        bytes.extend_from_slice(&self.digest);
        bytes.extend_from_slice(&[n, self.word.fractional, u8::from(self.word.signed)]);
        bytes.extend_from_slice(&(self.shape.source_intervals as u16).to_le_bytes());
        bytes.extend_from_slice(&(self.shape.padded_intervals as u16).to_le_bytes());
        bytes.extend_from_slice(&(self.shape.queries as u32).to_le_bytes());
        bytes.push(self.shape.boolean_outputs as u8);
        bytes.extend_from_slice(&(self.shape.vm_ops as u32).to_le_bytes());
        bytes.extend_from_slice(&self.mask.to_le_bytes()[..w]);
        for v in self.boundaries.iter() {
            bytes.extend_from_slice(&v.to_le_bytes()[..b]);
        }
        for (&ordinal, payload) in self.ordinals.iter().zip(self.coefficients.iter()) {
            bytes.extend_from_slice(&(ordinal as u16).to_le_bytes());
            for v in payload {
                bytes.extend_from_slice(&v.to_le_bytes()[..w]);
            }
        }
        for q in self.queries.iter() {
            bytes.push(q.bits);
            bytes.extend_from_slice(&q.shift.to_le_bytes()[..w]);
            bytes.extend_from_slice(&q.threshold.to_le_bytes()[..w]);
        }
        let mut offset = 0;
        for &length in &self.shape.program_lengths[..self.shape.boolean_outputs] {
            bytes.extend_from_slice(&(length as u32).to_le_bytes());
            for i in &self.instructions[offset..offset + length] {
                bytes.push(i.opcode as u8);
                bytes.extend_from_slice(&i.argument.to_le_bytes());
            }
            offset += length;
        }
        if bytes.len() != self.shape.reference_bytes {
            return Err(Error::Shape);
        }
        Ok(bytes)
    }
    pub(super) fn test_partition(&self) -> (&[u64], &[u64]) {
        (&self.boundaries, &self.ordinals)
    }
}

#[cfg(test)]
pub(super) fn emitter_owner_size() -> usize {
    std::mem::size_of::<Emitter>()
}
