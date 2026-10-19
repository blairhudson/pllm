//! Bounded scalar Protocol-3 compiler, not a semantic decoder activation.
//!
//! Public descriptors are immutable. Mask-dependent compilation is trusted-dealer
//! work only; neither `DealerPlan` nor its borrowed handoff is an evaluator key or
//! wire format. This module makes no production-security or model-identity claim.
#[cfg(test)]
mod accounting;
mod compile;
#[cfg(test)]
pub(crate) mod conformance;
mod fixture;
mod json;
#[cfg(test)]
mod tests;

pub use compile::{DealerHandoff, DealerPlan, Instruction, Opcode, PredicateQuery};
use json::Value;
use sha2::{Digest, Sha256};

pub const PROFILE: &str = "fusefss.source-derived-ring50-q12.v1";
pub const SYNTHETIC_PROFILE: &str = "fusefss.synthetic-conformance.v1";
pub const GELU_GATE: &str = "sigma-cpu-gelu.affine-q0-q12.identity.v1";
pub const GELU_DIGEST_HEX: &str =
    "5f1e888e7bdabe3cbad4b6bbf1abd6749c8dff0918c6f243916db071398e79d3";
pub const GELU_CANONICAL_BYTES: usize = 32900;
pub const SOURCE_LOCK_SHA256: &str =
    "8c42ba8f3ff7156fe87882dca089b8fb6c71b2739f8d761583e2ccc94c794609";
pub const MAX_DESCRIPTOR_BYTES: usize = 256 << 10;
pub const MAX_INSTANCE_BYTES: usize = 1 << 20;
pub const MAX_SOURCE_INTERVALS: usize = 512;
pub const MAX_BOOLEAN_OUTPUTS: usize = 4;
pub const MAX_EXPRESSION_NODES: usize = 8192;
pub const MAX_EXPRESSION_DEPTH: usize = 12;
pub const MAX_QUERIES: usize = 32768;
pub const MAX_VM_OPS: usize = 65536;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Allocation,
    DescriptorBytes,
    InstanceBytes,
    Encoding,
    Type,
    Range,
    Metadata,
    Partition,
    BooleanArity,
    ExpressionBudget,
    WorkBudget,
    UnsupportedProfile,
    UnsupportedPhi,
    DescriptorIdentity,
    Shape,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FuseFSS {self:?}")
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

pub(super) fn reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<()> {
    v.try_reserve(additional).map_err(|_| Error::Allocation)
}
pub(super) fn copy_bytes(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    reserve(&mut v, bytes.len())?;
    v.extend_from_slice(bytes);
    Ok(v)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArithmeticType {
    pub bits: u8,
    pub fractional: u8,
    pub signed: bool,
}

/// Descriptor-only work/layout. `reference_bytes` is the frozen diagnostic layout,
/// NOT backend key bytes, native residency, traffic, or an allocation reservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub source_intervals: usize,
    pub padded_intervals: usize,
    pub boolean_outputs: usize,
    pub queries: usize,
    pub vm_ops: usize,
    pub boolean_ands: usize,
    pub boolean_xors: usize,
    pub reference_bytes: usize,
    /// Exact lengths for each Boolean output's postfix program (at most four).
    pub program_lengths: [usize; MAX_BOOLEAN_OUTPUTS],
}
impl Shape {
    pub const DEGREE: usize = 1;
    pub const COEFFICIENTS_PER_INTERVAL: usize = 2;
    pub const HORNER_BEAVER_PRODUCTS: usize = 1;
    pub const HORNER_RING_ADDITIONS: usize = 1;
    pub const PHI_CONVERSIONS: usize = 0;
}

// Flat, typed source AST: no recursion or heap allocation per expression node.
#[derive(Clone, Copy)]
pub(super) enum Expr {
    Const(bool),
    Lt(u64),
    LowLt(u8, u64),
    SignedLt(i64),
    Msb(u64),
    Not(usize),
    Xor(usize, usize),
    And(usize, usize),
    Or(usize, usize),
}
pub(super) struct Piece {
    coefficients: [u64; 2],
    roots: [usize; MAX_BOOLEAN_OUTPUTS],
}

/// Original canonical bytes and source commitments are preserved without rewrite.
/// D8 descriptors are accepted only at the exact frozen GELU digest. Synthetic
/// descriptors retain the reference's full bounded Boolean expression language.
pub struct Descriptor {
    canonical: Vec<u8>,
    digest: [u8; 32],
    word: ArithmeticType,
    boundaries: Vec<u64>,
    pieces: Vec<Piece>,
    expressions: Vec<Expr>,
    shape: Shape,
}
impl Descriptor {
    /// Identity-first admission for the exact frozen GELU descriptor. Unlike the
    /// generic parser, untrusted alternate bytes are rejected before allocating
    /// a JSON DOM. The caller still accounts its input buffer separately. Use
    /// `gelu()` when no externally supplied descriptor bytes are needed.
    pub fn from_frozen_gelu_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != GELU_CANONICAL_BYTES {
            return Err(Error::DescriptorIdentity);
        }
        let digest: [u8; 32] = Sha256::new()
            .chain_update(b"FuseFSS-reference-descriptor-v1\0")
            .chain_update(bytes)
            .finalize()
            .into();
        if digest != fixture::GELU_DIGEST {
            return Err(Error::DescriptorIdentity);
        }
        Self::from_canonical_bytes(bytes)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self> {
        let obj = json::parse(bytes)?;
        obj.fields(&[
            "boundaries",
            "coefficient_fractional",
            "coefficient_order",
            "degree",
            "external_encoding",
            "gate",
            "input",
            "internal_rounding",
            "output",
            "partition_order",
            "phi",
            "pieces",
            "profile",
            "schema",
            "selector_semantics",
            "source_lock_sha256",
            "source_schedule",
        ])?;
        obj.get("schema")?
            .eq_string("pllm.fusefss.scalar-descriptor.v1")?;
        obj.get("external_encoding")?
            .eq_string("nearest-ties-even-input-and-weight")?;
        obj.get("internal_rounding")?
            .eq_string("signed-floor-only")?;
        obj.get("partition_order")?
            .eq_string("unsigned-canonical-half-open")?;
        if obj.get("phi")?.string()? != "identity" {
            return Err(Error::UnsupportedPhi);
        }
        if obj.get("degree")?.int()? != 1 {
            return Err(Error::Metadata);
        }
        let profile = obj.get("profile")?.string()?;
        if profile != PROFILE && profile != SYNTHETIC_PROFILE {
            return Err(Error::UnsupportedProfile);
        }
        let gate = obj.get("gate")?.string()?;
        if gate.is_empty() || gate.len() > 96 || !gate.is_ascii() {
            return Err(Error::Metadata);
        }
        let lock = obj.get("source_lock_sha256")?.string()?;
        if lock.len() != 64
            || !lock
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(Error::Metadata);
        }
        let input = obj.get("input")?;
        input.fields(&["bits", "domain", "fractional", "signed"])?;
        input.get("domain")?.eq_string("A")?;
        let bits = input.get("bits")?.uint()?;
        let fractional = input.get("fractional")?.uint()?;
        if !(1..=50).contains(&bits) || fractional >= bits {
            return Err(Error::Range);
        }
        let word = ArithmeticType {
            bits: bits as u8,
            fractional: fractional as u8,
            signed: input.get("signed")?.boolean()?,
        };
        let coeff_order = obj.get("coefficient_order")?.array()?;
        if coeff_order.len() != 2 {
            return Err(Error::Metadata);
        }
        coeff_order[0].eq_string("constant")?;
        coeff_order[1].eq_string("linear")?;
        let scales = obj.get("coefficient_fractional")?.array()?;
        if scales.len() != 2 || scales[0].uint()? != fractional || scales[1].uint()? != 0 {
            return Err(Error::Metadata);
        }
        obj.get("selector_semantics")?
            .eq_string(if profile == PROFILE {
                "floor-x-div-64-before-abs-clip255"
            } else {
                "descriptor-defined"
            })?;
        if profile == PROFILE {
            if word
                != (ArithmeticType {
                    bits: 50,
                    fractional: 12,
                    signed: true,
                })
                || gate != GELU_GATE
                || lock != SOURCE_LOCK_SHA256
            {
                return Err(Error::DescriptorIdentity);
            }
            // Exact frozen digest below validates every source_schedule field too.
        } else {
            obj.get("source_schedule")?.fields(&[])?;
        }
        let output = obj.get("output")?;
        output.fields(&["arithmetic", "boolean_arity"])?;
        output.get("arithmetic")?.eq_string("same-as-input")?;
        let arity =
            usize::try_from(output.get("boolean_arity")?.uint()?).map_err(|_| Error::Range)?;
        if arity > MAX_BOOLEAN_OUTPUTS || (profile == PROFILE && arity != 0) {
            return Err(Error::BooleanArity);
        }
        let n = 1u64 << bits;
        let src = obj.get("pieces")?.array()?;
        if src.is_empty() || src.len() > MAX_SOURCE_INTERVALS {
            return Err(Error::Partition);
        }
        let bounds = obj.get("boundaries")?.array()?;
        if bounds.len() != src.len() + 1 {
            return Err(Error::Partition);
        }
        let mut boundaries = Vec::new();
        reserve(&mut boundaries, bounds.len())?;
        for v in bounds {
            let v = v.uint()?;
            if v > n || boundaries.last().is_some_and(|p| *p >= v) {
                return Err(Error::Partition);
            }
            boundaries.push(v);
        }
        if boundaries[0] != 0 || boundaries.last() != Some(&n) {
            return Err(Error::Partition);
        }
        let mut expressions = Vec::new();
        let mut pieces = Vec::new();
        reserve(&mut pieces, src.len())?;
        for p in src {
            p.fields(&["booleans", "coefficients"])?;
            let c = p.get("coefficients")?.array()?;
            if c.len() != 2 {
                return Err(Error::Type);
            }
            let coefficients = [c[0].uint()?, c[1].uint()?];
            if coefficients.iter().any(|&c| c >= n) {
                return Err(Error::Range);
            }
            let booleans = p.get("booleans")?.array()?;
            if booleans.len() != arity {
                return Err(Error::BooleanArity);
            }
            let mut roots = [0; MAX_BOOLEAN_OUTPUTS];
            for (i, e) in booleans.iter().enumerate() {
                roots[i] = parse_expr(e, word.bits, 0, &mut expressions)?;
            }
            pieces.push(Piece {
                coefficients,
                roots,
            });
        }
        let canonical = copy_bytes(bytes)?;
        let digest: [u8; 32] = Sha256::new()
            .chain_update(b"FuseFSS-reference-descriptor-v1\0")
            .chain_update(bytes)
            .finalize()
            .into();
        if profile == PROFILE && digest != fixture::GELU_DIGEST {
            return Err(Error::DescriptorIdentity);
        }
        let mut descriptor = Self {
            canonical,
            digest,
            word,
            boundaries,
            pieces,
            expressions,
            shape: Shape {
                source_intervals: 0,
                padded_intervals: 0,
                boolean_outputs: arity,
                queries: 0,
                vm_ops: 0,
                boolean_ands: 0,
                boolean_xors: 0,
                reference_bytes: 0,
                program_lengths: [0; 4],
            },
        };
        descriptor.shape = compile::admission(&descriptor)?;
        Ok(descriptor)
    }
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn word(&self) -> ArithmeticType {
        self.word
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    /// Build fresh mask-dependent material in the trusted dealer only. Supplying
    /// randomness, freshness enforcement and complete live-owner admission belongs
    /// to the issuing backend, not this deterministic scalar compiler.
    pub fn compile(&self, mask: u64) -> Result<DealerPlan> {
        compile::compile(self, mask)
    }
    /// Exact frozen source-derived GELU descriptor. No upstream code is executed.
    pub fn gelu() -> Result<Self> {
        fixture::gelu()
    }
}

fn parse_expr(v: &Value, bits: u8, depth: usize, nodes: &mut Vec<Expr>) -> Result<usize> {
    if depth > MAX_EXPRESSION_DEPTH || nodes.len() >= MAX_EXPRESSION_NODES {
        return Err(Error::ExpressionBudget);
    }
    let a = v.array()?;
    if a.is_empty() || a.len() > 3 {
        return Err(Error::Type);
    }
    let tag = a[0].string()?;
    let n = 1u64 << bits;
    let e = match (tag, a.len()) {
        ("const", 2) => {
            let b = a[1].uint()?;
            if b > 1 {
                return Err(Error::Type);
            }
            Expr::Const(b != 0)
        }
        ("lt", 2) => {
            let t = a[1].uint()?;
            if t > n {
                return Err(Error::Range);
            }
            Expr::Lt(t)
        }
        ("low_lt", 3) => {
            let k = a[1].uint()?;
            if k == 0 || k > u64::from(bits) {
                return Err(Error::Range);
            }
            let t = a[2].uint()?;
            if t > 1u64 << k {
                return Err(Error::Range);
            }
            Expr::LowLt(k as u8, t)
        }
        ("slt", 2) => {
            let t = a[1].int()?;
            let h = (n / 2) as i64;
            if t < -h || t > h {
                return Err(Error::Range);
            }
            Expr::SignedLt(t)
        }
        ("msb", 2) => {
            let c = a[1].uint()?;
            if c >= n {
                return Err(Error::Range);
            }
            Expr::Msb(c)
        }
        ("not", 2) => Expr::Not(parse_expr(&a[1], bits, depth + 1, nodes)?),
        ("xor" | "and" | "or", 3) => {
            let l = parse_expr(&a[1], bits, depth + 1, nodes)?;
            let r = parse_expr(&a[2], bits, depth + 1, nodes)?;
            match tag {
                "xor" => Expr::Xor(l, r),
                "and" => Expr::And(l, r),
                _ => Expr::Or(l, r),
            }
        }
        _ => return Err(Error::Type),
    };
    if nodes.len() >= MAX_EXPRESSION_NODES {
        return Err(Error::ExpressionBudget);
    }
    reserve(nodes, 1)?;
    let index = nodes.len();
    nodes.push(e);
    Ok(index)
}
