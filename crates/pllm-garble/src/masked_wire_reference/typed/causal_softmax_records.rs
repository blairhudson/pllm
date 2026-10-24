//! Declaration hashes use the original length-prefixed digest, not a new codec.
//! Source labels and numeric pins are not signatures or imported authority.
use super::*;
use pllm_core::edge_contract::ConsumerUse;

pub(super) fn numeric(source: SourceIdentity, hashes: [[u8; 32]; 3]) -> [u8; 32] {
    digest(&[
        spec::DOMAIN,
        b"inverse.numeric.v1",
        spec::EQUATION,
        &source.checkpoint(),
        &source.numeric_profile(),
        &source.implementation(),
        crate::fixed_exp_tables::PAPER_SHA256.as_bytes(),
        crate::fixed_exp_tables::SOURCE_COMMIT.as_bytes(),
        &hashes[0],
        &hashes[1],
        &hashes[2],
        &[8, 0, 6, 50, 12],
        b"T[0]=0",
        b"T[z]=floor(2^18/z)",
    ])
}
pub(super) fn declaration(
    s: &CausalSoftmaxSpec,
    session: [u8; 32],
    hashes: [[u8; 32]; 3],
    uses: &[ConsumerUse],
) -> [u8; 32] {
    use sha2::Digest;
    let source = s.input.identity().source();
    let mut h = Sha256::new();
    let mut part = |bytes: &[u8]| {
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    };
    for value in [
        spec::DOMAIN,
        b"declaration.v1",
        spec::EQUATION,
        &source.checkpoint(),
        &source.numeric_profile(),
        &source.implementation(),
        &s.input.identity().plan(),
        &session,
        &s.operation,
        &s.next,
        crate::fixed_exp_tables::PAPER_SHA256.as_bytes(),
        crate::fixed_exp_tables::SOURCE_COMMIT.as_bytes(),
        &s.input.digest(),
        &s.output.digest(),
        &(s.input.shape().elements() as u64).to_le_bytes(),
        &[s.count],
        s.active_positions(),
        &(-spec::SCORE_LIMIT).to_le_bytes(),
        &spec::SCORE_LIMIT.to_le_bytes(),
        &[50, 12, 39, 14, 12, 6, 8, 6, 0, 50, 24, 12],
        &hashes[0],
        &hashes[1],
        &hashes[2],
        &numeric(source, hashes),
    ] {
        part(value);
    }
    for use_ in uses {
        part(&use_.digest());
    }
    h.finalize().into()
}
pub(super) fn stage(declaration: [u8; 32], stage: u16) -> [u8; 32] {
    digest(&[
        spec::DOMAIN,
        b"stage.v1",
        &declaration,
        &stage.to_le_bytes(),
    ])
}
/// Exact framed declaration length, no hashes or allocations.
#[cfg(test)]
pub(super) fn record_bytes(s: &CausalSoftmaxSpec) -> Result<usize, Error> {
    let lengths = [
        spec::DOMAIN.len(),
        b"declaration.v1".len(),
        spec::EQUATION.len(),
        32,
        32,
        32,
        32,
        32,
        32,
        32,
        crate::fixed_exp_tables::PAPER_SHA256.len(),
        crate::fixed_exp_tables::SOURCE_COMMIT.len(),
        32,
        32,
        8,
        1,
        usize::from(s.count),
        16,
        16,
        12,
        32,
        32,
        32,
        32,
    ];
    let mut bytes = 0;
    for length in lengths {
        bytes = add(bytes, add(8, length)?)?;
    }
    add(bytes, mul(s.outer_uses(), 8 + 32)?)
}
