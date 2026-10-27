//! Private, UNQUALIFIED conversion of an already-owned lexical Float node.
//!
//! Reuses `str::parse::<f64>()` and `f64::to_bits()` without numeric adaptation.
//! rust-toolchain.toml declares 1.85.1; the Cargo rust-version floor is not a pin.
//! No compiler/backend identity or parser execution has been verified here.
//!
//! This is a repeated pure borrowed conversion, NOT a one-shot root attempt.
//! Parser total work, auxiliary storage and whole-build stack are UNKNOWN. Even
//! a fixed-capacity std-parser storage audit would not bound whole-build stack.
//! Known lexeme bytes are not total parser work or a closed admission quote.
//! No guard adoption, budget/ledger, speculative prepayment or authority token.
//!
//! Input uses the existing UTF-8 scalar-value lexical profile, which rejects
//! lone surrogate escapes; paired escapes can decode through the existing path.
//! No blanket strict-decoding/domain equivalence or Python compatibility claim.
//! Infinity bits are NOT whole-pipeline infinity approval. Integer conversion,
//! codecs, Python object/error ownership, export, same-counter root funding and
//! native activation remain separate, unimplemented gates. C9 is unchanged.

use super::json::{LexicalNodeRef, LexicalNumberClass, LexicalValue};
use pllm_garble::shared_allocation::OwnedCharged;

/// Allocation-free refusal categories; never retain/format the parser error.
#[derive(Debug, Eq, PartialEq)]
pub(super) enum FloatError {
    MissingNode,
    NotFloat,
    BackendRejected,
}

/// Intentionally incomplete. None means UNKNOWN, not zero or qualified-empty.
/// Known bytes describe the borrowed lexeme only, not parser total resources.
pub(super) struct UnqualifiedResourceQuote {
    lexeme_bytes: usize,
    total_work: Option<u128>,
    total_auxiliary_bytes: Option<usize>,
    total_stack_bytes: Option<usize>,
}
impl UnqualifiedResourceQuote {
    pub(super) fn lexeme_bytes(&self) -> &usize {
        &self.lexeme_bytes
    }
    pub(super) fn total_work(&self) -> Option<&u128> {
        self.total_work.as_ref()
    }
    pub(super) fn total_auxiliary_bytes(&self) -> Option<&usize> {
        self.total_auxiliary_bytes.as_ref()
    }
    pub(super) fn total_stack_bytes(&self) -> Option<&usize> {
        self.total_stack_bytes.as_ref()
    }
}

/// Keeps the actual charged owner borrow AND its actual immutable lexical node.
/// No constructor accepts a caller string or manufactures a lexical node.
pub(super) struct FloatNode<'a> {
    owner: &'a OwnedCharged<LexicalValue>,
    node: LexicalNodeRef<'a>,
    resources: UnqualifiedResourceQuote,
}
impl<'a> FloatNode<'a> {
    pub(super) fn borrow(
        owner: &'a OwnedCharged<LexicalValue>,
        index: usize,
    ) -> Result<Self, FloatError> {
        let node = owner.payload().node(index).ok_or(FloatError::MissingNode)?;
        let bytes = match node.number() {
            Some((lexeme, LexicalNumberClass::Float)) => lexeme.len(),
            _ => return Err(FloatError::NotFloat),
        };
        // Slice/str length is already usize; no sum, narrowing or synthetic
        // LengthOverflow branch is needed. No parser work is attempted here.
        Ok(Self {
            owner,
            node,
            resources: UnqualifiedResourceQuote {
                lexeme_bytes: bytes,
                total_work: None,
                total_auxiliary_bytes: None,
                total_stack_bytes: None,
            },
        })
    }
    pub(super) fn owner(&self) -> &'a OwnedCharged<LexicalValue> {
        self.owner
    }
    pub(super) fn node(&self) -> &LexicalNodeRef<'a> {
        &self.node
    }
    pub(super) fn resources(&self) -> &UnqualifiedResourceQuote {
        &self.resources
    }
}

/// Bits are private and explicitly UNQUALIFIED, with original custody borrowed.
/// No qualified variant, promotion constructor, f64 export or detachable token.
pub(super) struct UnqualifiedFloat<'a> {
    node: FloatNode<'a>,
    bits: u64,
}
impl<'a> UnqualifiedFloat<'a> {
    pub(super) fn node(&self) -> &FloatNode<'a> {
        &self.node
    }
    pub(super) fn bits(&self) -> &u64 {
        &self.bits
    }
}

pub(super) fn parse_unqualified<'a>(node: FloatNode<'a>) -> Result<UnqualifiedFloat<'a>, FloatError> {
    let bits = match node.node.number() {
        Some((lexeme, LexicalNumberClass::Float)) => lexeme
            .parse::<f64>()
            .map_err(|_| FloatError::BackendRejected)?
            .to_bits(),
        _ => return Err(FloatError::NotFloat),
    };
    Ok(UnqualifiedFloat { node, bits })
}

#[cfg(test)]
#[path = "json_float_input/tests.rs"]
mod tests;
