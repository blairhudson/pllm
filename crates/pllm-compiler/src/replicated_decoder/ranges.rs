//! Public frozen range guards. No private observations affect construction.
use super::*;

#[path = "public_dot_metadata.rs"]
#[allow(dead_code)] // Inactive metadata leaf; no production consumer is attached.
mod public_dot_metadata;
pub(super) use public_dot_metadata::StoredRowDotCap;

/// Largest requested symmetric operand cap under a complete public row-L1 premise.
/// The triangle inequality bounds every product and prefix by cap * row_l1.
/// This pure calculation does not authenticate statistics, narrow a view, emit
/// a guard, admit a program, or account for post-dot bias or the full graph.
#[allow(dead_code)] // Deliberately not adopted by production consumers yet.
pub(super) fn public_dot_operand_cap(requested: i64, row_l1: u64) -> Result<i64, Error> {
    if requested < 0 {
        return Err(Error::Bounds);
    }
    if row_l1 == 0 {
        return Ok(requested);
    }
    // Native TruncAPr accepts [-2^62, 2^62); symmetry forgoes -2^62.
    let domain = (puma::INPUT_MAX_EXCLUSIVE - 1) as u128;
    let cap = (requested as u128).min(domain / u128::from(row_l1));
    i64::try_from(cap).map_err(|_| Error::Bounds)
}

#[cfg(test)]
#[path = "public_dot_operand_cap_tests.rs"]
mod public_dot_operand_cap_tests;

/// Maximal common symmetric cap at most requested, from public mathematical
/// premises only: for positive inner n and D = INPUT_MAX_EXCLUSIVE - 1,
/// n*C*C <= D. Conditional on actual integer operands satisfying |q_j|,|k_j|
/// <= C, every product, subset subtotal and prefix has magnitude <= n*C*C.
/// This proves no operand membership or provenance and supplies no guards,
/// view narrowing, certificates, authentication, admission or runtime adoption.
/// Maximality is among equal Q/K caps, not asymmetric pairs trading magnitudes.
/// Zero denotes only the singleton zero box; abstract widths admit no shape.
/// Products are raw encoded integers, before the existing complete-dot rescale.
/// Post-score scaling needs its own bound and cannot rescue an unsafe raw dot;
/// pre-rotary membership does not prove post-rotary membership. Bias, residual,
/// softmax, native constructor/resource checks and full graph fit remain separate.
#[allow(dead_code)] // Pure prerequisite; no production consumer attached.
pub(super) fn private_dot_operand_caps(
    requested: i64,
    inner: usize,
) -> Result<(i64, i64), Error> {
    if requested < 0 || inner == 0 {
        return Err(Error::Bounds);
    }
    let requested = u128::try_from(requested).map_err(|_| Error::Bounds)?;
    let inner = u128::try_from(inner).map_err(|_| Error::Bounds)?;
    let domain = puma::INPUT_MAX_EXCLUSIVE
        .checked_sub(1)
        .and_then(|value| u128::try_from(value).ok())
        .ok_or(Error::Bounds)?;
    let cap = requested.min((domain / inner).isqrt());
    let cap = i64::try_from(cap).map_err(|_| Error::Bounds)?;
    Ok((cap, cap))
}

#[cfg(test)]
#[path = "private_dot_operand_cap_tests.rs"]
mod private_dot_operand_cap_tests;

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
        let cap_energy = crate::replicated_decoder::checked_row_energy::cap_energy(
            width,
            self.checked_bound.ok_or(Error::Binding)?,
        )?;
        self.admission_stage = "checked_range_row_certificate";
        for (original, output) in words.chunks_exact(width).zip(safe.chunks_exact(width)) {
            if let Some(energy) = self.row_norm(&original, width) {
                // The scalar interval already proves cap_energy. Only retain
                // strict improvements, charging ordinary certificate metadata.
                if let Some(energy) =
                    crate::replicated_decoder::checked_row_energy::strict_improvement(
                        energy, cap_energy,
                    )
                {
                    self.record_row_energy(&output, energy)?;
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
        // retain the two-backing allowance with an exact descriptor pool.
        let tiles = words.len().div_ceil(tensor::MAX_TILE_LANES);
        self.admission_stage = "checked_range_view_metadata";
        self.charge_program_bytes(
            Words::span_pool_bytes(tiles)
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or(Error::Bounds)?,
        )?;
        let mut safe_words = Words::try_with_spans(tiles)?;
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
