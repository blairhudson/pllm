//! Public frozen range guards. No private observations affect construction.
use super::*;

impl Builder {
    /// Zero fallback cannot increase a certified complete row's exact energy,
    /// even on rejected requests. Match the entire ordered producer row; never
    /// infer a certificate for a subset, permutation, duplicate or mixed view.
    pub(super) fn checked_rows(&mut self, words: &Words, width: usize) -> Result<Words, Error> {
        if width == 0 || words.is_empty() || words.len() % width != 0 {
            return Err(Error::Shape);
        }
        let safe = self.checked_view(words)?;
        if safe == *words {
            return Ok(safe);
        }
        let cap =
            u128::try_from(self.checked_bound.ok_or(Error::Binding)?).map_err(|_| Error::Bounds)?;
        let cap_energy = cap
            .checked_mul(cap)
            .and_then(|square| square.checked_mul(width as u128))
            .ok_or(Error::Bounds)?;
        self.admission_stage = "checked_range_row_certificate";
        for (original, output) in words.chunks_exact(width).zip(safe.chunks_exact(width)) {
            if let Some(energy) = self.row_norm(&original, width) {
                // The scalar interval already proves cap_energy. Only retain
                // strict improvements, charging ordinary certificate metadata.
                if energy < cap_energy {
                    self.record_row_energy(&output, energy.min(cap_energy))?;
                }
            }
        }
        Ok(safe)
    }

    /// Replace only out-of-range private operands, retaining a request-wide
    /// sticky shared validity scalar. Publicly proved contained views are exact
    /// aliases, preserving existing row certificates and the static schedule.
    pub(super) fn checked_view(&mut self, words: &Words) -> Result<Words, Error> {
        self.admission_stage = "checked_range_domain";
        let Some(bound) = self.checked_bound else {
            return Ok(words.clone());
        };
        if bound < 0 || words.is_empty() {
            return Err(Error::Bounds);
        }
        let accepted = Interval::new(-bound, bound)?;
        let incoming = self.bounds(words)?;
        if incoming.lower() >= accepted.lower() && incoming.upper() <= accepted.upper() {
            return Ok(words.clone());
        }

        // Returned fragmented view metadata is retained alongside the graph's
        // ordinary construction-view allowance. Each tile contributes one span;
        // conservatively allow twice its rounded descriptor capacity.
        let tiles = words.len().div_ceil(tensor::MAX_TILE_LANES);
        self.admission_stage = "checked_range_view_metadata";
        self.charge_program_bytes(tiles.checked_mul(128).ok_or(Error::Bounds)?)?;
        let mut safe_words = Words::new();
        for tile in words.chunks(tensor::MAX_TILE_LANES) {
            let n = tile.len();
            self.admission_stage = "checked_range_input_gather";
            let input = self.gather(&tile)?;
            self.admission_stage = "checked_range_numeric_constructor";
            let program = tensor::Program::checked_range(n, self.nodes[input].bounds, accepted, 0)?;
            let digest = program.digest();
            // Charge and intern explicitly because CheckedRange has a distinct
            // two-output owner, not the ordinary Code::Tensor path. Inserting
            // before push also avoids double charging if push handles both kinds.
            let program = if let Some(program) = self.tensors.get(&digest) {
                program.clone()
            } else {
                self.admission_stage = "checked_range_program_metadata";
                self.charge_program_bytes(
                    program
                        .admitted_bytes()
                        .checked_add(1024)
                        .ok_or(Error::Bounds)?,
                )?;
                let program = Arc::new(program);
                self.tensors.insert(digest, program.clone());
                program
            };
            // This owner's public hull includes validity even when bound == 0.
            // Its two disjoint gathers below carry the stronger proven envelopes.
            self.admission_stage = "checked_range_output_owner";
            let owner = self.push(
                Code::CheckedRange { input, program },
                n.checked_mul(2).ok_or(Error::Bounds)?,
                Interval::new(-bound, bound.max(1))?,
            )?;
            let safe: Words = (0..n).map(|i| (owner, i)).collect();
            self.admission_stage = "checked_range_output_gathers";
            let safe = self.push(Code::Gather(safe), n, accepted)?;
            let valid: Words = (n..2 * n).map(|i| (owner, i)).collect();
            let valid = self.push(Code::Gather(valid), n, Interval::new(0, 1)?)?;
            self.admission_stage = "checked_range_validity_constructor";
            let valid_program = tensor::Program::all_valid(n)?;
            self.admission_stage = "checked_range_validity_owner";
            let valid = self.tensor(valid, valid_program)?;
            self.admission_stage = "checked_range_sticky_validity";
            self.validity = Some(match self.validity {
                Some(previous) => self.product(previous, valid, None, false)?,
                None => valid,
            });
            #[cfg(test)]
            if super::runtime::oracle_trace::enabled() {
                super::runtime::oracle_trace::range_tile(
                    bound,
                    owner,
                    n,
                    valid,
                    self.validity.unwrap(),
                );
            }
            safe_words.extend((0..n).map(|i| (safe, i)));
        }
        Ok(safe_words)
    }
}
