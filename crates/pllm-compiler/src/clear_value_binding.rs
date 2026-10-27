//! Private method/model-neutral custody, not an import/adoption API or executor.
//! The owned producer, not a cloned view or caller receipt, retains its actual
//! allocation and original budget/ledger. Semantic edge validation belongs to
//! trusted adapters; this seam checks only original owner and graph identities.
use crate::clear_input_owner::ClearInputs;

/// Implemented only inside trusted compiler adapters on actual retained owners.
/// Never public: external callers cannot assert source authority via this trait.
pub(crate) trait RetainedProducer {
    fn original(&self) -> &ClearInputs;
    fn operation_index(&self) -> usize;
}

pub(crate) struct ValueEdge<'a> {
    original: &'a ClearInputs,
    producer: usize,
    consumer: usize,
}
impl<'a> ValueEdge<'a> {
    pub(crate) fn new(original: &'a ClearInputs, producer: usize, consumer: usize) -> Option<Self> {
        let ops = original.operations();
        ops.get(producer)?;
        ops.get(consumer)?;
        Some(Self { original, producer, consumer })
    }
    pub(crate) fn original(&self) -> &'a ClearInputs { self.original }

    /// One-use ownership transfer. The implementation reads identities from the
    /// real retained producer, never from a separate caller-supplied owner label.
    pub(crate) fn bind<P: RetainedProducer>(self, producer: P) -> Result<BoundValue<'a, P>, ()> {
        if !std::ptr::eq(self.original, producer.original())
            || self.producer != producer.operation_index() {
            return Err(());
        }
        Ok(BoundValue { producer, edge: self })
    }
}

pub(crate) struct BoundValue<'a, P: RetainedProducer> {
    producer: P,
    edge: ValueEdge<'a>,
}
impl<P: RetainedProducer> BoundValue<'_, P> {
    pub(crate) fn producer(&self) -> &P { &self.producer }
    pub(crate) fn consumer(&self) -> usize { self.edge.consumer }
}
