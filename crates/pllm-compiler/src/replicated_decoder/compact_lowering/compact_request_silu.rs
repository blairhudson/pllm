//! Request-only closed guarded-SiLU preparation; no new source/row authority.
use super::*;
use crate::replicated_decoder::program::request_origin::compact_storage::ControllerAccounting;

impl<G: ControllerAccounting> CompactGraph<'_, '_, G> {
    pub(super) fn prepare_request_silu(
        &self,
        lanes: usize,
        input: Interval,
    ) -> Result<MapOwner, Error> {
        let context = self.request_admission.ok_or(Error::Binding)?;
        let mut attempt = context.attempt();
        context.check_open()?;
        let controls = std::mem::size_of::<MapOwner>()
            .checked_add(2 * std::mem::size_of::<Result<MapOwner, Error>>())
            .ok_or(Error::Bounds)?;
        let baseline = add_bytes(self.outside_owner_bytes, self.baseline()?)?;
        self.admit_total(add_bytes(baseline, controls)?)?;
        context.admit_phase(add_bytes(baseline, controls)?, 0)?;
        let peak = add_bytes(
            add_bytes(baseline, controls)?,
            PreparedTensorMap::request_silu_peak(context, lanes)?,
        )?;
        self.admit_total(peak)?;
        context.admit_phase(peak, 0)?;
        for (slot, op) in self.ops.iter().enumerate() {
            if op.prepared.as_ref().is_some_and(|p| {
                p.kind == MapKind::GuardedSilu && p.lanes == lanes && p.input == input
            }) {
                let prior = self.kernel(slot)?;
                prior.validate()?;
                if !prior.accepts(lanes, input)
                    || prior.one_loop_scope()
                        != Some(super::super::super::prepared_tensor_map::OneLoopScope::GuardedSilu)
                {
                    return Err(Error::Binding);
                }
                PreparedTensorMap::admit_request_silu_reuse(context, lanes, input)?;
                attempt.commit();
                return Ok(MapOwner {
                    kind: MapKind::GuardedSilu,
                    lanes,
                    input,
                    storage: MapStorage::Previous(slot),
                });
            }
        }
        let prepared = PreparedTensorMap::silu_request(context, lanes, input)?;
        // Equal numeric bytes never establish original Arc provenance. Keep the
        // existing partial-template collision refusal; no regenerated interning.
        for slot in 0..self.ops.len() {
            if self.ops[slot].prepared.is_none() {
                continue;
            }
            for old in self.kernel(slot)?.programs() {
                if prepared
                    .programs()
                    .any(|new| new.digest() == old.digest() && !Arc::ptr_eq(new, old))
                {
                    return Err(Error::Binding);
                }
            }
        }
        attempt.commit();
        Ok(MapOwner {
            kind: MapKind::GuardedSilu,
            lanes,
            input,
            storage: MapStorage::Owned(prepared),
        })
    }
}
