//! Consume construction aliases before moving original native invocation owners.
//! Data-only unpublished records: execution still requires complete request custody.
use super::*;

/// Private index into the genuine producing publication, not an Arc/digest key.
pub(in crate::replicated_decoder::program::compact_lowering) struct RowIndex {
    producer: usize,
    row: usize,
    start: usize,
    lanes: usize,
    reversed: bool,
}

struct Entry<'s> {
    producer: usize,
    row: usize,
    owner: Rc<OwnedRow<'s>>,
}

pub(in crate::replicated_decoder::program::compact_lowering) struct Conversion<'c, 's> {
    context: &'c EmissionContext<'s>,
    entries: Vec<Entry<'s>>,
}

// No context reference or Rc survives this record. Exact original Certified Arc
// and invocation coordinates MOVE; no factory, interning, or clone fallback.
struct Invocation {
    producer: usize,
    row: usize,
    predecessor: PublishedView,
    predecessor_native: SoftmaxPredecessor,
    prefix: usize,
    width: usize,
    gather: usize,
    tensor: usize,
    output_start: usize,
    keys: usize,
    zero: (usize, usize),
    native: Arc<Certified>,
}

pub(in crate::replicated_decoder::program::compact_lowering) struct Invocations {
    rows: Vec<Invocation>,
}

/// Read-only loan of one ORIGINAL invocation. Program sharing never merges rows.
#[derive(Clone, Copy)]
pub(in crate::replicated_decoder::program::compact_lowering) struct InvocationLoan<'r> {
    row: &'r Invocation,
}
impl<'r> InvocationLoan<'r> {
    pub(in crate::replicated_decoder::program::compact_lowering) fn predecessor(
        &self,
    ) -> &'r PublishedView {
        &self.row.predecessor
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn predecessor_programs(
        &self,
    ) -> impl Iterator<Item = &'r Arc<tensor::Program>> {
        self.row.predecessor_native.programs()
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn prefix_offset(&self) -> usize {
        self.row.prefix
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn native_width(&self) -> usize {
        self.row.width
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn gather_ordinal(&self) -> usize {
        self.row.gather
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn tensor_ordinal(&self) -> usize {
        self.row.tensor
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn output_start(&self) -> usize {
        self.row.output_start
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn keys(&self) -> usize {
        self.row.keys
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn zero_word(
        &self,
    ) -> (usize, usize) {
        self.row.zero
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn input_bounds(
        &self,
    ) -> Interval {
        self.row.native.input
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn output_bounds(
        &self,
    ) -> Interval {
        self.row.native.program.output_bounds()
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn mass_bound(&self) -> u64 {
        self.row.native.mass
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn certificate(
        &self,
    ) -> &'r Arc<Certified> {
        &self.row.native
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn program(
        &self,
    ) -> &'r Arc<tensor::Program> {
        &self.row.native.program
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn input_word(
        &self,
        index: usize,
    ) -> Result<(usize, usize), Error> {
        if index >= self.row.width {
            return Err(Error::Shape);
        }
        self.row
            .predecessor
            .word(add_bytes(self.row.prefix, index)?)
    }
}
impl RowIndex {
    pub(in crate::replicated_decoder::program::compact_lowering) fn coordinates(
        &self,
    ) -> (usize, usize, usize, usize, bool) {
        (
            self.producer,
            self.row,
            self.start,
            self.lanes,
            self.reversed,
        )
    }
}

impl<'c, 's> Conversion<'c, 's> {
    pub(in crate::replicated_decoder::program::compact_lowering) fn new(
        context: &'c EmissionContext<'s>,
        ops: &[PublishedOp<'s>],
    ) -> Result<Self, Error> {
        context.check_open()?;
        context
            .plan()
            .prepay_compiler_metadata(
                &context.reservation(),
                u64::try_from(ops.len().checked_mul(8).ok_or(Error::Bounds)?)
                    .map_err(|_| Error::Bounds)?,
            )
            .map_err(|_| Error::Bounds)?;
        let count = ops.iter().try_fold(0usize, |n, op| {
            add_bytes(
                n,
                match &op.kind {
                    PublishedKind::AttentionSoftmax(rows) => rows.len(),
                    _ => 0,
                },
            )
        })?;
        let retained = add_bytes(size_of::<Invocations>(), array_bytes::<Invocation>(count)?)?;
        let transition = add_bytes(
            retained,
            add_bytes(
                add_bytes(
                    array_bytes::<Entry<'s>>(count)?,
                    array_bytes::<(usize, Rows<'s>)>(count)?,
                )?,
                size_of::<Self>()
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(2 * size_of::<Result<Invocations, Error>>()))
                    .ok_or(Error::Bounds)?,
            )?,
        )?;
        // Old Rc/native/vector backing is already retained by context. Reserve
        // ALL destination and mapping overlap above it BEFORE allocating.
        context.admit_phase(transition, 0)?;
        #[cfg(test)]
        super::super::frozen_region::owning::conversion_probe::nested(transition)?;
        // Entries, group scratch and result/move controls remain live while
        // Values allocates its destination. Retain the whole transition until
        // the owning stage retires, not just the final invocation vector.
        context.retain(transition)?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        if entries.capacity() != count {
            return Err(Error::Bounds);
        }
        for (producer, op) in ops.iter().enumerate() {
            if let PublishedKind::AttentionSoftmax(rows) = &op.kind {
                for (row, owner) in rows.native.iter().enumerate() {
                    if !std::ptr::eq(owner.context, context) {
                        return Err(Error::Binding);
                    }
                    owner.context.check_open()?;
                    entries.push(Entry {
                        producer,
                        row,
                        owner: Rc::clone(owner),
                    });
                }
            }
        }
        Ok(Self { context, entries })
    }

    pub(in crate::replicated_decoder::program::compact_lowering) fn index(
        &self,
        view: RowView<'s>,
    ) -> Result<RowIndex, Error> {
        self.context.check_open()?;
        self.context
            .plan()
            .prepay_compiler_metadata(
                &self.context.reservation(),
                u64::try_from(self.entries.len().checked_mul(4).ok_or(Error::Bounds)?)
                    .map_err(|_| Error::Bounds)?,
            )
            .map_err(|_| Error::Bounds)?;
        let actual = self
            .entries
            .iter()
            .find(|entry| Rc::ptr_eq(&entry.owner, &view.owner))
            .ok_or(Error::Binding)?;
        if !std::ptr::eq(view.owner.context, self.context)
            || view
                .start
                .checked_add(view.lanes)
                .is_none_or(|end| end > view.owner.keys)
        {
            return Err(Error::Binding);
        }
        let index = RowIndex {
            producer: actual.producer,
            row: actual.row,
            start: view.start,
            lanes: view.lanes,
            reversed: view.reversed,
        };
        // Consuming the view retires its construction Rc alias now.
        drop(view);
        Ok(index)
    }

    /// Call only AFTER every downstream RowView has become an index. Unexpected
    /// aliases fail closed; they are never copied/reconstructed to force success.
    pub(in crate::replicated_decoder::program::compact_lowering) fn finish(
        self,
        groups: Vec<(usize, Rows<'s>)>,
    ) -> Result<Invocations, Error> {
        let Self { context, entries } = self;
        let count = entries.len();
        let mut destination = Vec::new();
        destination
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        if destination.capacity() != count {
            return Err(Error::Bounds);
        }
        // Lookup aliases are retired before unwrapping the ORIGINAL row owners.
        drop(entries);
        for (producer, rows) in groups {
            for (row, owner) in rows.native.into_iter().enumerate() {
                if !std::ptr::eq(owner.context, context) {
                    return Err(Error::Binding);
                }
                owner.context.check_open()?;
                let OwnedRow {
                    predecessor,
                    predecessor_native,
                    prefix,
                    width,
                    gather,
                    tensor,
                    output_start,
                    keys,
                    zero,
                    native,
                    context: _,
                } = Rc::try_unwrap(owner).map_err(|_| Error::Binding)?;
                destination.push(Invocation {
                    producer,
                    row,
                    predecessor,
                    predecessor_native,
                    prefix,
                    width,
                    gather,
                    tensor,
                    output_start,
                    keys,
                    zero,
                    native,
                });
            }
        }
        if destination.len() != count {
            return Err(Error::Binding);
        }
        fn send_sync<T: Send + Sync>() {}
        send_sync::<Invocations>();
        Ok(Invocations { rows: destination })
    }
}

impl Invocations {
    pub(in crate::replicated_decoder::program::compact_lowering) fn row(
        &self,
        producer: usize,
        row: usize,
    ) -> Result<InvocationLoan<'_>, Error> {
        let row = self
            .rows
            .iter()
            .find(|actual| actual.producer == producer && actual.row == row)
            .ok_or(Error::Binding)?;
        if row.width == 0
            || row.native.width != row.width
            || row.native.program.lanes() != row.width
            || row.keys < row.width
            || row.gather.checked_add(1) != Some(row.tensor)
            || row
                .prefix
                .checked_add(row.width)
                .is_none_or(|end| end > row.predecessor.lanes)
        {
            return Err(Error::Binding);
        }
        Ok(InvocationLoan { row })
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn programs(
        &self,
    ) -> impl Iterator<Item = &Arc<tensor::Program>> {
        self.rows.iter().map(|row| &row.native.program)
    }
    pub(in crate::replicated_decoder::program::compact_lowering) fn word(
        &self,
        index: &RowIndex,
        lane: usize,
    ) -> Result<(usize, usize), Error> {
        let actual = self
            .rows
            .iter()
            .find(|row| row.producer == index.producer && row.row == index.row)
            .ok_or(Error::Binding)?;
        if lane >= index.lanes {
            return Err(Error::Shape);
        }
        let offset = index
            .start
            .checked_add(if index.reversed {
                index.lanes - 1 - lane
            } else {
                lane
            })
            .ok_or(Error::Bounds)?;
        if offset >= actual.keys {
            return Err(Error::Binding);
        }
        Ok(if offset < actual.width {
            (actual.tensor, offset)
        } else {
            actual.zero
        })
    }
}
