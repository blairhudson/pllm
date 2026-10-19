//! Attention-only complete-vector energy ledger. These certificates never enter
//! general RMSNorm input admission or replace the original scalar/RoPE checks.
use super::*;
use puma::multiply::norm::RowEnergyProof;

impl Builder {
    pub(super) fn attention_energy(&self, words: &Words, width: usize) -> Option<u128> {
        if width == 0 || words.is_empty() || words.len() % width != 0 {
            return None;
        }
        words.chunks_exact(width).try_fold(0, |bound, row| {
            let energy = self
                .attention_energies
                .iter()
                .find_map(|(original, energy)| {
                    (original.len() == width && original.iter().eq(row.iter())).then_some(*energy)
                })?;
            Some(bound.max(energy))
        })
    }

    pub(super) fn record_attention_energy(
        &mut self,
        words: &Words,
        energy: u128,
    ) -> Result<(), Error> {
        let bytes = words
            .metadata_bytes()
            .checked_add(std::mem::size_of::<(Words, u128)>())
            .ok_or(Error::Bounds)?;
        let allocated = self
            .allocated
            .checked_add(bytes)
            .filter(|bytes| *bytes <= MAX_PROGRAM_BYTES)
            .ok_or(Error::Bounds)?;
        self.attention_energies
            .try_reserve_exact(1)
            .map_err(|_| Error::Allocation)?;
        self.attention_energies.push((words.clone(), energy));
        self.allocated = allocated;
        Ok(())
    }

    pub(super) fn projection_energies(
        &mut self,
        input: &View,
        output: &View,
        weight: &SourceView,
        bias: Option<&SourceView>,
    ) -> Result<(), Error> {
        let Some(norms) = self.public_norms.get(&weight.owner) else {
            return Ok(());
        };
        let (rows, columns, width) = norms.geometry();
        if weight.shape != [rows, columns]
            || input.shape.last() != Some(&columns)
            || output.shape.last() != Some(&rows)
            || output.words.len() / rows != input.words.len() / columns
        {
            return Err(Error::Binding);
        }
        let Some(input_energy) = self.row_norm(&input.words, columns) else {
            return Ok(());
        };
        for row in output.words.chunks_exact(rows) {
            for start in (0..rows).step_by(width) {
                let count = width.min(rows - start);
                let certificate = self
                    .public_norms
                    .get(&weight.owner)
                    .and_then(|norms| norms.block(start, count))
                    .ok_or(Error::Binding)?;
                let mut energy = norm_bounds::linear_energy(input_energy, certificate, 18)?;
                if let Some(bias) = bias {
                    if bias.shape != [rows] {
                        return Err(Error::Shape);
                    }
                    let bias = self
                        .public_norms
                        .get(&bias.owner)
                        .and_then(|norms| norms.block(start, count))
                        .ok_or(Error::Binding)?;
                    energy = norm_bounds::sum_energy(energy, bias.frobenius_squared)?;
                }
                self.record_attention_energy(&row.slice(start..start + count), energy)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_dot(
        &self,
        row: &Words,
        weight: &MatrixView<'_>,
        start: usize,
        count: usize,
        inner: usize,
        left: usize,
        right: usize,
    ) -> Result<Option<RowEnergyProof>, Error> {
        let MatrixView::Values(keys) = weight else {
            return Ok(None);
        };
        let Some(left_energy) = self.attention_energy(row, inner) else {
            return Ok(None);
        };
        // MatrixView::Values gathers physical right[inner,columns] from intact
        // logical key rows. Never use physical right-row energy here.
        let begin = start.checked_mul(inner).ok_or(Error::Bounds)?;
        let end = start
            .checked_add(count)
            .and_then(|n| n.checked_mul(inner))
            .ok_or(Error::Bounds)?;
        if end > keys.words.len() {
            return Err(Error::Shape);
        }
        let Some(right_energy) = self.attention_energy(&keys.words.slice(begin..end), inner) else {
            return Ok(None);
        };
        let bounds = InputBounds {
            left: self.nodes[left].bounds,
            right: self.nodes[right].bounds,
        };
        Ok(Some(RowEnergyProof::new(
            inner,
            bounds,
            left_energy,
            right_energy,
        )?))
    }
}

#[cfg(test)]
#[path = "attention_energy_tests.rs"]
mod tests;
