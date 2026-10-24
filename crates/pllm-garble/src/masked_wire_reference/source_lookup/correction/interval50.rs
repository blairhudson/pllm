//! Private paired-interval extension of ORIGINAL D42; not an arbitrary scan API.
use super::super::interval50::{IntervalIssueSeal, IntervalKeyLane};
use super::*;

#[cfg_attr(not(test), allow(dead_code))] // Private D43 native qualification; whole parent not activated.
impl ValidatedDescriptor {
    pub(in crate::masked_wire_reference::source_lookup) fn interval50(
        output_bits: u8,
    ) -> Result<Self, Error> {
        if !matches!(output_bits, 13 | 50) {
            return Err(Error::Domain);
        }
        Ok(Self(CorrectionDescriptor {
            wire_spec: Spec {
                index_bits: 50,
                output_bits,
                lanes: 1,
                boundary: Boundary::Reconstructed,
            },
        }))
    }
}
#[cfg_attr(not(test), allow(dead_code))] // Output-specific key/finish tickets remain private until parent issuance.
impl<C: AllocationContext> CorrectionBuildFor<C> {
    pub(in crate::masked_wire_reference::source_lookup) fn push_interval_keypair(
        &mut self,
        lane: IntervalKeyLane,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if self.descriptor.wire_spec
            != ValidatedDescriptor::interval50(lane.output_bits())?
                .0
                .wire_spec
        {
            self.failed = true;
            return Err(Error::Context);
        }
        // Non-Clone output-specific ticket was born only after the ONE actual
        // GPU50 key pair. Reuse original four draws, partial wipe and cursor.
        self.push_index(0, fill)
    }
    pub(in crate::masked_wire_reference::source_lookup) fn finish_interval(
        self,
        seal: IntervalIssueSeal,
        fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> CorrectionResult<C> {
        let (spec, binding, context) = seal.consume();
        if spec
            != ValidatedDescriptor::interval50(spec.output_bits)?
                .0
                .wire_spec
        {
            return Err(Error::Context);
        }
        self.finish_fields(spec, binding, context, fill)
    }
}
