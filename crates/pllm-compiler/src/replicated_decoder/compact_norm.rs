//! Bounded RMSNorm publication through the original storage-neutral driver.
//! Original whole-input guard order and compiler-local sticky custody. Prefix
//! exports still require a genuine incoming issuer, not declarations or digests.
use super::*;
use graph::driver::{self, RmsNormSink};

pub(super) struct NormPublication<'a> {
    pub(super) kind: PublishedKind<'a>,
    pub(super) bounds: Interval,
    pub(super) nodes: usize,
    pub(super) work: WorkBudget,
    pub(super) rows_first: usize,
}

/// Borrowed ordered coordinates, not expanded Words or a new proof owner.
#[derive(Clone, Copy)]
struct RowWords<'s> {
    view: &'s PublishedView,
    start: usize,
    lanes: usize,
    safe: bool,
}

struct Template<'a> {
    gamma: &'a SourceTensor,
    input_energy: Option<u128>,
    output_energy: u128,
    program: Arc<tensor::Program>,
    owns_program: bool,
}

struct NormSink<'s, 'g, 'a> {
    graph: &'s CompactGraph<'g, 'a>,
    input: &'s PublishedView,
    template: Option<Template<'a>>,
    rows: usize,
    gamma: Option<&'a SourceTensor>,
    checked: Option<compact_validity::CheckedRows<'a>>,
    prepared: bool,
    bounds: Option<Interval>,
    energy: Option<u128>,
    work: WorkBudget,
}

// Pending guard owners coexist with the RMS constructor and its return value.
// Restore the existing shared control ledger on every error/return path.
struct Pending<'s> {
    cell: &'s std::cell::Cell<usize>,
    before: usize,
}
impl Drop for Pending<'_> {
    fn drop(&mut self) {
        self.cell.set(self.before);
    }
}

impl NormSink<'_, '_, '_> {
    fn prepare(&mut self, gamma: &SourceTensor, width: usize) -> Result<(), Error> {
        if self.prepared {
            return Ok(());
        }
        let (checked, bounds, energy, work) =
            compact_validity::prepare(self.graph, self.input, width, gamma)?;
        self.graph.contraction_control_bytes.set(add_bytes(
            self.graph.contraction_control_bytes.get(),
            checked.as_ref().map_or(Ok(0), |rows| rows.baseline())?,
        )?);
        self.checked = checked;
        self.bounds = Some(bounds);
        self.energy = energy;
        self.work = work;
        self.prepared = true;
        Ok(())
    }
}

/// Caller/driver return overlap, in addition to the existing native quote and
/// retained graph baseline. No native inventory is reconstructed here.
pub(super) fn driver_slots() -> Result<usize, Error> {
    let mut bytes = std::mem::size_of::<NormSink<'_, '_, '_>>();
    for slot in [
        std::mem::size_of::<RowWords<'_>>(),
        2 * std::mem::size_of::<Option<RowWords<'_>>>(),
        2 * std::mem::size_of::<Result<RowWords<'_>, Error>>(),
        2 * std::mem::size_of::<Result<NormPublication<'_>, Error>>(),
        2 * std::mem::size_of::<Result<usize, Error>>(),
        std::mem::size_of::<AdmissionSite>(),
    ] {
        bytes = add_bytes(bytes, slot)?;
    }
    Ok(bytes)
}

/// Shared with the all-before-storage transition quote. Keep the actual original
/// RMS helper's return/control overlap, not just its native construction quote.
pub(super) fn factory_slots() -> Result<usize, Error> {
    use std::mem::size_of;
    let mut bytes = driver_slots()?;
    for slot in [
        2 * size_of::<Result<(Arc<tensor::Program>, u128, bool), Error>>(),
        2 * size_of::<Result<(tensor::Program, Option<u128>), Error>>(),
        size_of::<tensor::Construction>(),
        size_of::<tensor::ConstructionQuote>(),
        size_of::<Result<tensor::ConstructionQuote, tensor::ConstructionQuoteError>>(),
        size_of::<tensor::Resources>(),
        size_of::<Result<tensor::Resources, Error>>(),
    ] {
        bytes = add_bytes(bytes, slot)?;
    }
    Ok(bytes)
}

impl<'s> RmsNormSink for NormSink<'s, '_, '_> {
    type Words = RowWords<'s>;
    type Owner = ();
    type Output = usize;

    fn checked(&self) -> bool {
        self.graph.imports.checked_bound.is_some()
    }

    fn checked_rows(&mut self, words: &Self::Words, width: usize) -> Result<Self::Words, Error> {
        if !std::ptr::eq(words.view, self.input)
            || words.start != 0
            || words.lanes != self.input.lanes
            || words.safe
        {
            return Err(Error::Binding);
        }
        self.prepare(self.gamma.ok_or(Error::Binding)?, width)?;
        let checked = self.checked.as_ref().ok_or(Error::Binding)?;
        checked.row(self.graph.imports, words.view, 0, width)?;
        Ok(RowWords {
            safe: true,
            ..*words
        })
    }

    fn rows(words: &Self::Words, width: usize) -> impl Iterator<Item = Self::Words> + '_ {
        (0..words.lanes / width).map(move |row| RowWords {
            view: words.view,
            start: words.start + row * width,
            lanes: width,
            safe: words.safe,
        })
    }

    fn rms_norm_row(
        &mut self,
        row: &Self::Words,
        width: usize,
        gamma: &SourceTensor,
    ) -> Result<(), Error> {
        if !std::ptr::eq(row.view, self.input)
            || row.lanes != width
            || row.start != self.rows.checked_mul(width).ok_or(Error::Bounds)?
            || row
                .start
                .checked_add(width)
                .is_none_or(|n| n > self.input.lanes)
        {
            return Err(Error::Binding);
        }
        if self.template.is_none() {
            // Reborrow the actual validated owner. Equal digests or copied gamma
            // bytes do not authorize a different SourceTensor lifetime/owner.
            let gamma = self
                .graph
                .sources
                .values()
                .find(|source| std::ptr::eq(*source, gamma))
                .ok_or(Error::Binding)?;
            // The closed publication subset retains one common bound/energy
            // contract across its rows. row_norm resolves only existing exact
            // general-row lineage; attention-only block energy never enters it.
            self.prepare(gamma, width)?;
            let input_energy = self.energy;
            if row.safe {
                let proof = self.checked.as_ref().ok_or(Error::Binding)?.row(
                    self.graph.imports,
                    row.view,
                    row.start,
                    width,
                )?;
                if Some(proof.bounds()) != self.bounds {
                    return Err(Error::Binding);
                }
                // Resolve the actual safe owner, not a relabelled original view.
                proof.word(width - 1)?;
            } else if self.checked.is_some() {
                return Err(Error::Binding);
            }
            let (program, output_energy, owns_program) = self.graph.prepare_rms_norm(
                width,
                self.bounds.ok_or(Error::Binding)?,
                gamma,
                input_energy,
            )?;
            self.template = Some(Template {
                gamma,
                input_energy,
                output_energy,
                program,
                owns_program,
            });
        } else if !std::ptr::eq(self.template.as_ref().ok_or(Error::Binding)?.gamma, gamma) {
            return Err(Error::Binding);
        }
        if row.safe {
            self.checked.as_ref().ok_or(Error::Binding)?.row(
                self.graph.imports,
                row.view,
                row.start,
                width,
            )?;
        }
        self.rows = self.rows.checked_add(1).ok_or(Error::Bounds)?;
        Ok(())
    }

    fn append(output: &mut usize, _: (), width: usize) {
        // Geometry is validated before driver entry; rows partition MAX_VALUES.
        *output += width;
    }

    // Expanded oracle owns range-event lane arrays. Compact publication emits no
    // execution trace or checked-producer claim.
    #[cfg(test)]
    fn range_begin(&self, _: &AdmissionSite, _: &str, _: &[usize], _: &Self::Words) {}

    #[cfg(test)]
    fn range_end(&self) {}
}

impl<'a> CompactGraph<'_, 'a> {
    pub(super) fn lower_rms_norm(
        &self,
        op: &pllm_models::ModelOperation,
        input: &PublishedView,
        shape: &[usize],
    ) -> Result<NormPublication<'a>, Error> {
        self.validate_view(input)?;
        if op.operator != Op::RmsNorm
            || op.inputs.len() != 1
            || shape != input.logical.dimensions()
            || shape.len() != 3
            || shape[0] != 1
            || shape[2] == 0
            || input.lanes % shape[2] != 0
        {
            return Err(Error::Shape);
        }
        let _pending = Pending {
            cell: &self.contraction_control_bytes,
            before: self.contraction_control_bytes.get(),
        };
        let mut sink = NormSink {
            graph: self,
            input,
            template: None,
            rows: 0,
            // Borrow-only lookup; the shared driver remains the validation owner.
            gamma: op
                .attributes
                .get("weight")
                .and_then(|v| v.as_str())
                .and_then(|key| self.sources.get(key)),
            checked: None,
            prepared: false,
            bounds: None,
            energy: None,
            work: WorkBudget::default(),
        };
        let words = RowWords {
            view: input,
            start: 0,
            lanes: input.lanes,
            safe: false,
        };
        let site = AdmissionSite {
            step: Some(self.cursor),
            operator: Some(op.operator),
            layer: op.layer,
            ..AdmissionSite::default()
        };
        let lanes = driver::rms_norm(&mut sink, op, shape, Some(&words), self.sources, &site)?;
        if lanes != input.lanes || sink.rows != shape[1] {
            return Err(Error::Binding);
        }
        let template = sink.template.ok_or(Error::Binding)?;
        let bounds = template.program.output_bounds();
        let native_work = template
            .program
            .work_budget()
            .checked_mul(sink.rows as u64)?;
        let guard_work = sink
            .checked
            .as_ref()
            .map_or(WorkBudget::default(), |rows| rows.work);
        let work = native_work.checked_add(guard_work)?;
        if work != sink.work {
            return Err(Error::Binding);
        }
        let rows_first = sink
            .checked
            .as_ref()
            .map_or(self.next, |rows| rows.rows_first());
        let nodes = add_bytes(
            rows_first.checked_sub(self.next).ok_or(Error::Binding)?,
            sink.rows.checked_mul(2).ok_or(Error::Bounds)?,
        )?;
        Ok(NormPublication {
            kind: PublishedKind::RmsNorm {
                gamma: template.gamma,
                input_energy: template.input_energy,
                output_energy: template.output_energy,
                program: template.program,
                owns_program: template.owns_program,
                checked: sink.checked,
                row_bounds: sink.bounds.ok_or(Error::Binding)?,
            },
            bounds,
            nodes,
            work,
            rows_first,
        })
    }
}
