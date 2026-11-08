//! Request-only rescale preparation. Projection/bias/source authority stays in
//! the original ContractionSink; reuse shares backing but never invocation work.
use super::*;
use crate::replicated_decoder::program::request_origin::compact_storage::ControllerAccounting;

impl<G: ControllerAccounting> CompactGraph<'_, '_, G> {
    pub(super) fn prepare_request_rescale(
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
        // Reserve graph/import/caller controls before querying native work or
        // allocating. Existing retained owners get no digest/retired-peak credit.
        self.admit_total(add_bytes(baseline, controls)?)?;
        context.admit_phase(add_bytes(baseline, controls)?, 0)?;
        let peak = add_bytes(
            add_bytes(baseline, controls)?,
            PreparedTensorMap::request_rescale_peak(context, lanes)?,
        )?;
        self.admit_total(peak)?;
        context.admit_phase(peak, 0)?;
        for (slot, op) in self.ops.iter().enumerate() {
            if op.prepared.as_ref().is_some_and(|p| {
                p.kind == MapKind::Rescale18 && p.lanes == lanes && p.input == input
            }) {
                let prior = self.kernel(slot)?;
                prior.validate()?;
                if !prior.accepts(lanes, input)
                    || prior.one_loop_scope()
                        != Some(super::super::super::prepared_tensor_map::OneLoopScope::Rescale18)
                {
                    return Err(Error::Binding);
                }
                PreparedTensorMap::admit_request_rescale_reuse(context, lanes, input, 18)?;
                attempt.commit();
                return Ok(MapOwner {
                    kind: MapKind::Rescale18,
                    lanes,
                    input,
                    storage: MapStorage::Previous(slot),
                });
            }
        }
        let prepared = PreparedTensorMap::rescale_request(context, lanes, input, 18)?;
        // Preserve existing partial-template collision refusal. Never intern a
        // regenerated equal digest or substitute a separately allocated Arc.
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
            kind: MapKind::Rescale18,
            lanes,
            input,
            storage: MapStorage::Owned(prepared),
        })
    }
}
