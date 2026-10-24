//! Public D36 geometry and source declaration. Not issuance/range authority.
use super::*;
use pllm_core::edge_contract::{EdgeIdentity, Numeric, Shape};

pub(super) const DOMAIN: &[u8] = b"pllm.sigma.causal-softmax-row.v1";
pub(super) const EQUATION: &[u8] = b"M=max(active X);E=nExp(M-X);D=sum(E);Z=floor(D/64);R=floor(2^18/Z);P=floor(E*R/4096);inactive=0";
pub(super) const SCORE_LIMIT: i128 = 1i128 << 37;
pub(super) const MAX_USES: usize = 40;

/// At most three PUBLIC row positions; active positions are nonempty, unique
/// and in increasing public position order. Inactive values do not participate
/// in the private arithmetic. All declared original scores still pass admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalSoftmaxSpec {
    pub(super) input: EdgeDescriptor,
    pub(super) output: EdgeDescriptor,
    pub(super) active: [u8; 3],
    pub(super) count: u8,
    pub(super) operation: [u8; 32],
    pub(super) next: [u8; 32],
}

/// Deterministic logical admission, not CPU instructions, timing or RSS.
/// Caller subtracts the COMPLETE aggregate from its enclosing work ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CausalSoftmaxWorkLimit(pub u128);

impl CausalSoftmaxSpec {
    pub fn checked(
        input: EdgeDescriptor,
        output: EdgeDescriptor,
        active: &[u8],
        operation: [u8; 32],
        next: [u8; 32],
    ) -> Result<Self, Error> {
        let width = input.shape().elements();
        if !(1..=3).contains(&width) || !(1..=3).contains(&active.len()) {
            return Err(Error::Domain);
        }
        input
            .identity()
            .first_tensor_index()
            .checked_add(width as u64)
            .ok_or(Error::Resource)?;
        if input.numeric()
            != Numeric::new(50, 12, Interpretation::Signed).map_err(|_| Error::Domain)?
            || output.numeric() != input.numeric()
            || output.shape() != input.shape()
            || input.representation() != Representation::CommonAdditiveMasked
            || output.representation()
                != if active.len() == 1 {
                    Representation::Public
                } else {
                    Representation::CommonAdditiveMasked
                }
            || input.identity().source() != output.identity().source()
            || input.identity().plan() != output.identity().plan()
            || input.identity().first_tensor_index() != output.identity().first_tensor_index()
            || input.identity().wire() == output.identity().wire()
            || input.weight_visibility().is_some()
            || output.weight_visibility().is_some()
        {
            return Err(Error::Domain);
        }
        for (i, &position) in active.iter().enumerate() {
            if usize::from(position) >= width || i > 0 && active[i - 1] >= position {
                return Err(Error::Domain);
            }
        }
        let mut positions = [0; 3];
        positions[..active.len()].copy_from_slice(active);
        Ok(Self {
            input,
            output,
            active: positions,
            count: active.len() as u8,
            operation,
            next,
        })
    }
    pub fn input_descriptor(&self) -> EdgeDescriptor {
        self.input
    }
    pub fn output_descriptor(&self) -> EdgeDescriptor {
        self.output
    }
    pub fn active_positions(&self) -> &[u8] {
        &self.active[..usize::from(self.count)]
    }
    pub fn input(&self) -> EdgeDescriptor {
        self.input
    }
    pub fn output(&self) -> EdgeDescriptor {
        self.output
    }
    pub fn exchanges(&self) -> usize {
        if self.count == 1 {
            0
        } else {
            16 + 2 * usize::from(self.count)
        }
    }
    pub fn outer_uses(&self) -> usize {
        if self.count == 1 {
            2
        } else {
            13 * usize::from(self.count) + 1
        }
    }

    pub(super) fn check_original<C: AllocationContext>(
        &self,
        dealer: &LocalDealerFor<C>,
        input: &CausalSoftmaxInput,
    ) -> Result<(), Error> {
        if self.input.identity().source() != dealer.source
            || self.input.identity().plan() != dealer.plan
            || !std::ptr::eq(input.budget(), dealer.budget.budget())
        {
            return Err(Error::Context);
        }
        let bytes = add(
            size_of::<CausalSoftmaxInput>(),
            mul(input.payload().capacity(), size_of::<i64>())?,
        )?;
        if input.charged_bytes() < bytes || input.payload().len() != self.input.shape().elements() {
            return Err(Error::Resource);
        }
        for &value in input.payload().iter() {
            if i128::from(value) < -SCORE_LIMIT || i128::from(value) >= SCORE_LIMIT {
                return Err(Error::Domain);
            }
        }
        Ok(())
    }

    /// Internal virtual wires are derived from fixed public slots. This labels
    /// the original private lane mapping; it does not create an input capability.
    pub(super) fn wire(
        &self,
        slot: u16,
        lanes: usize,
        bits: u8,
        scale: u8,
        kind: Interpretation,
    ) -> Result<EdgeDescriptor, Error> {
        let id = self.input.identity();
        let wire = digest(&[
            DOMAIN,
            b"wire.v1",
            &id.wire(),
            &self.operation,
            &slot.to_le_bytes(),
        ]);
        EdgeDescriptor::new(
            EdgeIdentity::new(
                id.source(),
                id.plan(),
                id.producer(),
                wire,
                id.first_tensor_index(),
            ),
            Shape::new(&[lanes]).map_err(|_| Error::Domain)?,
            Numeric::new(bits, scale, kind).map_err(|_| Error::Domain)?,
            if kind == Interpretation::Predicate {
                Representation::CommonXorMasked
            } else {
                Representation::CommonAdditiveMasked
            },
            None,
        )
        .map_err(|_| Error::Domain)
    }
}
