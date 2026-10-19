//! Public frozen range guards. No private observations affect construction.
use super::*;

impl Builder {
    /// Replace only out-of-range private operands, retaining a request-wide
    /// sticky shared validity scalar. Publicly proved contained views are exact
    /// aliases, preserving existing row certificates and the static schedule.
    pub(super) fn checked_view(&mut self, words: &Words) -> Result<Words, Error> {
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
        self.allocated = self
            .allocated
            .checked_add(tiles.checked_mul(128).ok_or(Error::Bounds)?)
            .filter(|n| *n <= MAX_PROGRAM_BYTES)
            .ok_or(Error::Bounds)?;
        let mut safe_words = Words::new();
        for tile in words.chunks(tensor::MAX_TILE_LANES) {
            let n = tile.len();
            let input = self.gather(&tile)?;
            let program = tensor::Program::checked_range(n, self.nodes[input].bounds, accepted, 0)?;
            let digest = program.digest();
            // Charge and intern explicitly because CheckedRange has a distinct
            // two-output owner, not the ordinary Code::Tensor path. Inserting
            // before push also avoids double charging if push handles both kinds.
            let program = if let Some(program) = self.tensors.get(&digest) {
                program.clone()
            } else {
                self.allocated = self
                    .allocated
                    .checked_add(program.admitted_bytes())
                    .and_then(|n| n.checked_add(1024))
                    .filter(|n| *n <= MAX_PROGRAM_BYTES)
                    .ok_or(Error::Bounds)?;
                let program = Arc::new(program);
                self.tensors.insert(digest, program.clone());
                program
            };
            // This owner's public hull includes validity even when bound == 0.
            // Its two disjoint gathers below carry the stronger proven envelopes.
            let owner = self.push(
                Code::CheckedRange { input, program },
                n.checked_mul(2).ok_or(Error::Bounds)?,
                Interval::new(-bound, bound.max(1))?,
            )?;
            let safe: Words = (0..n).map(|i| (owner, i)).collect();
            let safe = self.push(Code::Gather(safe), n, accepted)?;
            let valid: Words = (n..2 * n).map(|i| (owner, i)).collect();
            let valid = self.push(Code::Gather(valid), n, Interval::new(0, 1)?)?;
            let valid = self.tensor(valid, tensor::Program::all_valid(n)?)?;
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
