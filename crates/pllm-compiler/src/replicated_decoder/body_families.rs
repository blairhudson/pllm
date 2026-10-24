//! Original constructors and public loop geometry for every layer family.
use super::*;

type RotaryCoefficientRow = [i64; tensor::MAX_TILE_LANES / 2];
// Two full destination arrays, independent of the currently active row width.
const ROTARY_COEFFICIENT_BYTES: usize = 2 * size_of::<RotaryCoefficientRow>();
// Each explicit zero-array initializer and its destination may overlap. This
// inventories those four array value slots, not generated stack/RSS usage.
const ROTARY_INITIALIZATION_BYTES: usize = 2 * ROTARY_COEFFICIENT_BYTES;

impl Builder {
    /// Native owner quote plus independently retained caller buffers. The live
    /// Builder census is added by proof_preflight; neither borrowed slice bytes
    /// nor a template already interned in that census earns a subtraction.
    /// This does not replace post-construction capacity/work/numeric admission.
    fn native_construction_preflight(
        &mut self,
        construction: tensor::Construction,
        caller_buffer_bytes: usize,
    ) -> Result<(), Error> {
        let result = (|| {
            self.proof_admission_live()?;
            let quote = construction.quote().map_err(|error| match error {
                tensor::ConstructionQuoteError::Geometry(error) => error,
                // The native coefficient generator does not constrain actual
                // Vec capacities. Requested lengths cannot authorize backing.
                tensor::ConstructionQuoteError::UncheckedRotaryCoefficientCapacity => Error::Bounds,
            })?;
            self.proof_preflight(
                caller_buffer_bytes
                    .checked_add(quote.peak_with_borrowed_public_bytes)
                    .ok_or(Error::Bounds)?,
            )
        })();
        if result.is_err() {
            self.proof_admission_retired = true;
        }
        result
    }

    pub fn binary(
        &mut self,
        left: &Layout,
        right: &Layout,
        multiply: bool,
    ) -> Result<Layout, Error> {
        self.binary_with_temporary(left, right, multiply, 0)
    }
    fn binary_with_temporary(
        &mut self,
        left: &Layout,
        right: &Layout,
        multiply: bool,
        temporary: usize,
    ) -> Result<Layout, Error> {
        if left.len() != right.len() {
            return Err(Error::Shape);
        }
        let slices = Self::proof_row_scratch(left)?
            .checked_add(Self::proof_row_scratch(right)?)
            .and_then(|n| n.checked_add(temporary))
            .ok_or(Error::Bounds)?;
        let mut regions = self.region_buffer(left.len().div_ceil(TILE), slices)?;
        let mut work = WorkBudget::default();
        for offset in (0..left.len()).step_by(TILE) {
            let len = TILE.min(left.len() - offset);
            let bounds = InputBounds {
                left: self.bounds(&left.slice(offset, len)?)?,
                right: self.bounds(&right.slice(offset, len)?)?,
            };
            let range = if multiply {
                work = work.checked_add(WorkBudget::product(len, true)?)?;
                bounds.output()?
            } else {
                multiply::arithmetic::Affine {
                    left: 1,
                    right: 1,
                    constant: 0,
                }
                .output_bounds(bounds)?
            };
            regions.push(Region {
                offset,
                len,
                bounds: range,
            });
        }
        self.push_with_temporary(
            Op::Binary {
                left: left.clone(),
                right: right.clone(),
                multiply,
            },
            left.len(),
            regions,
            work,
            if multiply {
                left.len().div_ceil(TILE) as u64
            } else {
                0
            },
            temporary,
        )
    }
    pub fn residual(
        &mut self,
        left: &Layout,
        right: &Layout,
        width: usize,
    ) -> Result<Layout, Error> {
        if width == 0 || left.len() % width != 0 {
            return Err(Error::Shape);
        }
        let mut energies = Vec::new();
        let rows = left.len() / width;
        let temporary = rows.checked_mul(size_of::<u128>()).ok_or(Error::Bounds)?;
        let slices = Self::proof_row_scratch(left)?
            .checked_add(Self::proof_row_scratch(right)?)
            .ok_or(Error::Bounds)?;
        self.proof_preflight(temporary.checked_add(slices).ok_or(Error::Bounds)?)?;
        if energies.try_reserve_exact(rows).is_err() {
            self.proof_admission_retired = true;
            return Err(Error::Allocation);
        }
        if energies.capacity() != rows {
            self.proof_admission_retired = true;
            return Err(Error::Bounds);
        }
        for offset in (0..left.len()).step_by(width) {
            let a = left.slice(offset, width)?;
            let b = right.slice(offset, width)?;
            let energy = |this: &Self, row: &Layout| -> Result<u128, Error> {
                if let Some(p) = this.proof(row).and_then(|p| p.energy) {
                    return Ok(p);
                }
                (0..row.len()).try_fold(0u128, |sum, i| {
                    let r = this.bounds(&row.slice(i, 1)?)?;
                    let m = u128::from(r.lower().unsigned_abs().max(r.upper().unsigned_abs()));
                    sum.checked_add(m * m).ok_or(Error::Bounds)
                })
            };
            let root = |n: u128| {
                let r = n.isqrt();
                r + u128::from(r * r != n)
            };
            let radius = root(energy(self, &a)?)
                .checked_add(root(energy(self, &b)?))
                .ok_or(Error::Bounds)?;
            energies.push(radius.checked_mul(radius).ok_or(Error::Bounds)?);
        }
        let output = self.binary_with_temporary(left, right, false, temporary)?;
        let result = (|| {
            let temporary = temporary
                .checked_add(output.metadata_bytes())
                .and_then(|n| n.checked_add(64))
                .ok_or(Error::Bounds)?;
            for (row, squared) in energies.iter().copied().enumerate() {
                self.record_proof(
                    Proof {
                        key: Self::proof_row_key(&output, row * width, width)
                            .ok_or(Error::Binding)?,
                        energy: Some(squared),
                        mass: None,
                    },
                    temporary,
                )?;
            }
            Ok(output)
        })();
        if result.is_err() {
            self.proof_admission_retired = true;
        }
        result
    }
    /// Retain actual native row allocations without rebuilding/interning them.
    /// This method confers NO source or certificate authority: the sealed caller
    /// independently validates and retains its original execution owner.
    pub(in crate::replicated_decoder) fn retained_native_rows<'a>(
        &mut self,
        input: &Layout,
        width: usize,
        rows: impl ExactSizeIterator<Item = &'a Arc<tensor::Program>>,
    ) -> Result<Layout, Error> {
        let result = (|| {
            self.proof_admission_live()?;
            let capacity = rows.len();
            self.proof_preflight(
                capacity
                    .checked_mul(2 * size_of::<Arc<tensor::Program>>())
                    .ok_or(Error::Bounds)?,
            )?;
            let mut programs = Vec::new();
            programs
                .try_reserve_exact(capacity)
                .map_err(|_| Error::Allocation)?;
            if programs.capacity() != capacity {
                return Err(Error::Allocation);
            }
            programs.extend(rows.cloned());
            self.tensors(input, width, programs, vec![], 0)
        })();
        if result.is_err() {
            self.proof_admission_retired = true
        }
        result
    }
    fn tensors(
        &mut self,
        input: &Layout,
        stride: usize,
        programs: Vec<Arc<tensor::Program>>,
        certificates: Vec<(usize, Option<u128>, Option<u64>)>,
        enclosing_temporary: usize,
    ) -> Result<Layout, Error> {
        if stride == 0 || input.len().div_ceil(stride) != programs.len() {
            return Err(Error::Shape);
        }
        let mut regions = Vec::new();
        let temporary = certificates
            .capacity()
            .checked_mul(size_of::<(usize, Option<u128>, Option<u64>)>())
            .and_then(|n| n.checked_add(enclosing_temporary))
            .ok_or(Error::Bounds)?;
        // Input Vec backing remains live while destination boxes are formed.
        let construction = programs
            .capacity()
            .checked_add(programs.len())
            .and_then(|n| n.checked_mul(size_of::<Arc<tensor::Program>>()))
            .and_then(|n| {
                programs
                    .len()
                    .checked_mul(2 * size_of::<Region>())?
                    .checked_add(n)
            })
            .and_then(|n| n.checked_add(temporary))
            .and_then(|n| n.checked_add(input.metadata_bytes()))
            .and_then(|n| n.checked_add(256))
            .ok_or(Error::Bounds)?;
        self.proof_preflight(construction)?;
        if regions.try_reserve_exact(programs.len()).is_err() {
            self.proof_admission_retired = true;
            return Err(Error::Allocation);
        }
        if regions.capacity() != programs.len() {
            self.proof_admission_retired = true;
            return Err(Error::Bounds);
        }
        let mut work = WorkBudget::default();
        let mut advances = 0u64;
        for (row, program) in programs.iter().enumerate() {
            let offset = row * stride;
            if program.lanes() > stride.min(input.len() - offset)
                || program.output_lanes() > stride.min(input.len() - offset)
                || program.validity_lanes().is_some()
            {
                return Err(Error::Shape);
            }
            work = work.checked_add(program.work_budget())?;
            advances = advances
                .checked_add(program.instructions() as u64)
                .ok_or(Error::Bounds)?;
            regions.push(Region {
                offset,
                len: program.output_lanes(),
                bounds: program.output_bounds(),
            });
        }
        let output = self.push_with_temporary(
            Op::Tensor {
                input: input.clone(),
                stride,
                programs: programs.into_boxed_slice(),
            },
            input.len(),
            regions,
            work,
            advances,
            temporary,
        )?;
        let result = (|| {
            let temporary = temporary
                .checked_add(output.metadata_bytes())
                .and_then(|n| n.checked_add(input.metadata_bytes()))
                .and_then(|n| n.checked_add(128))
                .ok_or(Error::Bounds)?;
            for (row, energy, mass) in certificates.iter().copied() {
                let width = match self.ops.last().ok_or(Error::Binding)? {
                    Op::Tensor { programs, .. } => {
                        programs.get(row).ok_or(Error::Binding)?.output_lanes()
                    }
                    _ => return Err(Error::Binding),
                };
                let offset = row.checked_mul(stride).ok_or(Error::Bounds)?;
                self.record_proof(
                    Proof {
                        key: Self::proof_row_key(&output, offset, width).ok_or(Error::Binding)?,
                        energy,
                        mass,
                    },
                    temporary,
                )?;
            }
            Ok(output)
        })();
        if result.is_err() {
            self.proof_admission_retired = true;
        }
        result
    }
    fn chunks(
        &mut self,
        input: &Layout,
        construction: impl Fn(usize) -> tensor::Construction,
        make: impl Fn(usize, Interval) -> Result<tensor::Program, Error>,
    ) -> Result<Layout, Error> {
        // PublicMatrix may already have pushed its unscaled owner before this
        // family is called. Native constructor failures must burn that Builder.
        let result = (|| {
            let mut programs = self.new_template_list(input, input.len().div_ceil(TILE), 0)?;
            for offset in (0..input.len()).step_by(TILE) {
                let n = TILE.min(input.len() - offset);
                self.native_construction_preflight(
                    construction(n),
                    Self::template_call_temporary(input, &programs, 0)?,
                )?;
                let program = make(n, self.bounds(&input.slice(offset, n)?)?)?;
                let temporary = Self::template_call_temporary(input, &programs, 0)?;
                programs.push(self.intern_with_temporary(program, temporary)?);
            }
            self.tensors(input, TILE, programs, vec![], 0)
        })();
        if result.is_err() {
            self.proof_admission_retired = true;
        }
        result
    }
    pub fn rescale(&mut self, input: &Layout) -> Result<Layout, Error> {
        self.chunks(
            input,
            |lanes| tensor::Construction::Builder { lanes },
            |n, bounds| {
                let mut b = tensor::Builder::new(n, bounds)?;
                let output = b.rescale(b.input(), 18)?;
                b.finish(output)
            },
        )
    }
    /// Deliberately includes masked slots. Zero values do not elide any rescale.
    pub fn attention_scale(&mut self, input: &Layout, head_dim: usize) -> Result<Layout, Error> {
        if head_dim == 0 {
            return Err(Error::Shape);
        }
        let coefficient = ((1.0_f32 / (head_dim as f32).sqrt()) * ONE as f32) as i64;
        self.chunks(
            input,
            |lanes| tensor::Construction::Builder { lanes },
            |n, bounds| {
                let mut b = tensor::Builder::new(n, bounds)?;
                let raw = b.affine(
                    b.input(),
                    None,
                    multiply::arithmetic::Affine {
                        left: coefficient,
                        right: 0,
                        constant: 0,
                    },
                )?;
                let output = b.rescale(raw, 18)?;
                b.finish(output)
            },
        )
    }
    pub fn silu(&mut self, input: &Layout) -> Result<Layout, Error> {
        self.chunks(
            input,
            |lanes| tensor::Construction::GuardedSilu { lanes },
            |n, bounds| tensor::Program::activation_guarded(tensor::Activation::Silu, n, bounds),
        )
    }
    /// All complete membership tiles and sticky products precede normalization.
    pub fn checked(
        &mut self,
        input: &Layout,
        bound: i64,
        previous: Option<&Layout>,
    ) -> Result<(Layout, Option<Layout>), Error> {
        self.checked_with_temporary(input, bound, previous, 0)
    }
    fn checked_with_temporary(
        &mut self,
        input: &Layout,
        bound: i64,
        previous: Option<&Layout>,
        enclosing_temporary: usize,
    ) -> Result<(Layout, Option<Layout>), Error> {
        if bound < 0 {
            return Err(Error::Bounds);
        }
        if let Some(p) = previous {
            let b = self.bounds(p)?;
            if p.len() != 1 || b.lower() < 0 || b.upper() > 1 {
                return Err(Error::Binding);
            }
        }
        let incoming = self.bounds(input)?;
        let accepted = Interval::new(-bound, bound)?;
        if incoming.lower() >= accepted.lower() && incoming.upper() <= accepted.upper() {
            return Ok((input.clone(), previous.cloned()));
        }
        let result = (|| {
            let previous_bytes = previous
                .map_or(0, |p| p.metadata_bytes() + 64)
                .checked_add(enclosing_temporary)
                .ok_or(Error::Bounds)?;
            let tiles = input.len().div_ceil(tensor::MAX_TILE_LANES);
            let mut checked = self.new_template_list(input, tiles, previous_bytes)?;
            let mut valid = self.new_template_list(
                input,
                tiles,
                previous_bytes
                    .checked_add(checked.capacity() * size_of::<Arc<tensor::Program>>())
                    .ok_or(Error::Bounds)?,
            )?;
            let mut work = WorkBudget::default();
            let mut advances = 0u64;
            for (tile, offset) in (0..input.len()).step_by(tensor::MAX_TILE_LANES).enumerate() {
                let n = tensor::MAX_TILE_LANES.min(input.len() - offset);
                let temporary = Self::template_call_temporary(
                    input,
                    &checked,
                    previous_bytes
                        .checked_add(valid.capacity() * size_of::<Arc<tensor::Program>>())
                        .ok_or(Error::Bounds)?,
                )?;
                self.native_construction_preflight(
                    tensor::Construction::CheckedRange { lanes: n },
                    temporary,
                )?;
                let program = tensor::Program::checked_range(
                    n,
                    self.bounds(&input.slice(offset, n)?)?,
                    accepted,
                    0,
                )?;
                let a = self.intern_with_temporary(program, temporary)?;
                // The just-interned guard is now in the live census, even
                // though it has not yet been pushed into the local list.
                self.native_construction_preflight(
                    tensor::Construction::AllValid { lanes: n },
                    temporary,
                )?;
                let program = tensor::Program::all_valid(n)?;
                let b = self.intern_with_temporary(program, temporary)?;
                work = work
                    .checked_add(a.work_budget())?
                    .checked_add(b.work_budget())?;
                advances += a.instructions() as u64 + b.instructions() as u64;
                if tile > 0 || previous.is_some() {
                    work = work.checked_add(WorkBudget::product(1, false)?)?;
                    advances += 1;
                }
                checked.push(a);
                valid.push(b);
            }
            let n = input.len();
            let temporary = Self::template_call_temporary(
                input,
                &checked,
                previous_bytes
                    .checked_add(valid.capacity() * size_of::<Arc<tensor::Program>>())
                    .ok_or(Error::Bounds)?,
            )?;
            let mut regions = self.region_buffer(2, temporary)?;
            regions.push(Region {
                offset: 0,
                len: n,
                bounds: accepted,
            });
            regions.push(Region {
                offset: n,
                len: 1,
                bounds: Interval::new(0, 1)?,
            });
            // Both list Vecs coexist with their boxed destinations during freezing.
            self.proof_preflight(
                temporary
                    .checked_add(
                        tiles
                            .checked_mul(2 * size_of::<Arc<tensor::Program>>())
                            .ok_or(Error::Bounds)?,
                    )
                    .and_then(|n| n.checked_add(regions.capacity() * size_of::<Region>()))
                    .and_then(|n| n.checked_add(256))
                    .ok_or(Error::Bounds)?,
            )?;
            let output = self.push_with_temporary(
                Op::Check {
                    input: input.clone(),
                    previous: previous.cloned(),
                    accepted,
                    checked: checked.into_boxed_slice(),
                    valid: valid.into_boxed_slice(),
                },
                n.checked_add(1).ok_or(Error::Bounds)?,
                regions,
                work,
                advances,
                previous_bytes,
            )?;
            self.proof_preflight(
                Self::proof_row_scratch(&output)?
                    .checked_add(previous_bytes)
                    .ok_or(Error::Bounds)?,
            )?;
            let safe = self.checked_return_slice(&output, 0, n)?;
            let valid = self.checked_return_slice(&output, n, 1)?;
            Ok((safe, Some(valid)))
        })();
        if result.is_err() {
            self.proof_admission_retired = true;
        }
        result
    }
    /// Production calls the original Layout allocator unchanged. The local
    /// test seam observes/fails each returned-slice attempt immediately before
    /// that call, not individual allocator internals or infallible Arc OOM.
    fn checked_return_slice(
        &mut self,
        output: &Layout,
        start: usize,
        len: usize,
    ) -> Result<Layout, Error> {
        #[cfg(test)]
        {
            let attempt = self.checked_return_slice_attempts;
            self.checked_return_slice_attempts += 1;
            if self.checked_return_slice_fail_at == Some(attempt) {
                return Err(Error::Allocation);
            }
        }
        output.slice(start, len)
    }
    /// Propagate only existing complete ordered input-row energy. Coordinatewise
    /// safe-zero cannot increase energy even when sticky request validity is 0.
    /// Mass and attention-only certificates are intentionally not transferred.
    pub fn checked_rows(
        &mut self,
        input: &Layout,
        width: usize,
        bound: i64,
        previous: Option<&Layout>,
    ) -> Result<(Layout, Option<Layout>), Error> {
        self.checked_rows_with_temporary(input, width, bound, previous, 0)
    }
    fn checked_rows_with_temporary(
        &mut self,
        input: &Layout,
        width: usize,
        bound: i64,
        previous: Option<&Layout>,
        enclosing_temporary: usize,
    ) -> Result<(Layout, Option<Layout>), Error> {
        if width == 0 || input.len() % width != 0 {
            return Err(Error::Shape);
        }
        if bound < 0 {
            return Err(Error::Bounds);
        }
        let incoming = self.bounds(input)?;
        let result = (|| {
            let (safe, valid) =
                self.checked_with_temporary(input, bound, previous, enclosing_temporary)?;
            if incoming.lower() >= -bound && incoming.upper() <= bound {
                return Ok((safe, valid));
            }
            let cap = crate::replicated_decoder::checked_row_energy::cap_energy(width, bound)?;
            // Returned layouts own backing not yet retained by a subsequent Op.
            let temporary = safe
                .metadata_bytes()
                .checked_add(64)
                .and_then(|n| {
                    valid.as_ref().map_or(Some(n), |v| {
                        v.metadata_bytes().checked_add(64)?.checked_add(n)
                    })
                })
                .and_then(|n| n.checked_add(enclosing_temporary))
                .ok_or(Error::Bounds)?;
            for offset in (0..input.len()).step_by(width) {
                let Some(energy) = self.proof_at(input, offset, width).and_then(|p| p.energy)
                else {
                    continue;
                };
                if let Some(energy) =
                    crate::replicated_decoder::checked_row_energy::strict_improvement(energy, cap)
                {
                    let key = Self::proof_row_key(&safe, offset, width).ok_or(Error::Binding)?;
                    self.record_proof(
                        Proof {
                            key,
                            energy: Some(energy),
                            mass: None,
                        },
                        temporary,
                    )?;
                }
            }
            Ok((safe, valid))
        })();
        if result.is_err() {
            self.proof_admission_retired = true;
        }
        result
    }
    pub fn rms(
        &mut self,
        input: &Layout,
        width: usize,
        gamma: &[i64],
        checked: Option<i64>,
        previous: Option<&Layout>,
    ) -> Result<(Layout, Option<Layout>), Error> {
        if width == 0 || input.len() % width != 0 || gamma.len() != width {
            return Err(Error::Shape);
        }
        let result = (|| {
            // This API borrows slice elements, not the unknown enclosing Vec
            // capacity/header. Keep those elements live through guards, native
            // construction, interning and certificate publication separately.
            let gamma_bytes = gamma
                .len()
                .checked_mul(size_of::<i64>())
                .ok_or(Error::Bounds)?;
            let (safe, valid) = match checked {
                Some(bound) => {
                    self.checked_rows_with_temporary(input, width, bound, previous, gamma_bytes)?
                }
                None => (input.clone(), previous.cloned()),
            };
            let mut programs = Vec::new();
            let mut certificates = Vec::new();
            let temporary = valid
                .as_ref()
                .map_or(0, |v| v.metadata_bytes() + 64)
                .checked_add(gamma_bytes)
                .ok_or(Error::Bounds)?;
            self.proof_producer_buffers(&safe, width, &mut programs, &mut certificates, temporary)?;
            for (row, offset) in (0..safe.len()).step_by(width).enumerate() {
                self.proof_producer_row_preflight(&safe, &programs, &certificates, temporary)?;
                let input = safe.slice(offset, width)?;
                let bounds = self.bounds(&input)?;
                let input_norm = self.proof(&input).and_then(|p| p.energy);
                let construction = match input_norm {
                    Some(_) => tensor::Construction::RmsNormWithInputNorm {
                        rows: 1,
                        width,
                        gamma_len: gamma.len(),
                    },
                    None => tensor::Construction::RmsNormCertified {
                        rows: 1,
                        width,
                        gamma_len: gamma.len(),
                    },
                };
                // gamma is already included in `temporary` for every other
                // boundary. Here the quote itself prices its borrowed elements.
                let native_temporary = Self::proof_producer_temporary(
                    &safe,
                    &programs,
                    &certificates,
                    valid.as_ref().map_or(0, |v| v.metadata_bytes() + 64),
                )?;
                self.native_construction_preflight(construction, native_temporary)?;
                let (program, squared) = match input_norm {
                    Some(norm) => {
                        tensor::Program::rms_norm_with_input_norm(1, width, bounds, gamma, norm)?
                    }
                    None => tensor::Program::rms_norm_certified(1, width, bounds, gamma)?,
                };
                let call_temporary =
                    Self::proof_producer_temporary(&safe, &programs, &certificates, temporary)?;
                programs.push(self.intern_with_temporary(program, call_temporary)?);
                certificates.push((row, Some(squared), None));
            }
            Ok((
                self.tensors(&safe, width, programs, certificates, temporary)?,
                valid,
            ))
        })();
        if result.is_err() {
            self.proof_admission_retired = true;
        }
        result
    }
    /// Allocation-free coefficient generation into pre-admitted fixed buffers;
    /// native rotary_half retains its own authoritative construction quote.
    pub fn rope(
        &mut self,
        input: &Layout,
        width: usize,
        sequence: usize,
        position: usize,
        theta: f32,
    ) -> Result<Layout, Error> {
        self.rope_with_fill(
            input,
            width,
            sequence,
            position,
            theta,
            tensor::Program::rotary_coefficients_into,
        )
    }
    /// Same production path with an observable fill boundary for focused tests.
    /// No test-only allocator or alternate native program constructor is used.
    fn rope_with_fill(
        &mut self,
        input: &Layout,
        width: usize,
        sequence: usize,
        position: usize,
        theta: f32,
        fill: impl Fn(f32, u32, usize, &mut [i64], &mut [i64]) -> Result<(), Error>,
    ) -> Result<Layout, Error> {
        let result = (|| {
            self.proof_admission_live()?;
            if width == 0
                || width > tensor::MAX_TILE_LANES
                || width % 2 != 0
                || sequence == 0
                || input.len() % width != 0
                || input.len() / width % sequence != 0
            {
                return Err(Error::Shape);
            }
            input.validate(&self.registers)?;
            // Every sequence position occurs: rows is a positive multiple of
            // sequence. Validate its maximum before list/array construction.
            let last_position = position.checked_add(sequence - 1).ok_or(Error::Bounds)?;
            if !theta.is_finite() || theta < 1.0 || last_position > 512 {
                return Err(Error::Bounds);
            }
            // This existing gate prices the future list's exact capacity, input
            // view scratch, both FULL coefficient arrays and initialization
            // overlap before the list allocation or either array initializer.
            let mut programs =
                self.new_template_list(input, input.len() / width, ROTARY_INITIALIZATION_BYTES)?;
            let mut cosine: RotaryCoefficientRow = [0; tensor::MAX_TILE_LANES / 2];
            let mut sine: RotaryCoefficientRow = [0; tensor::MAX_TILE_LANES / 2];
            let half = width / 2;
            for (row, offset) in (0..input.len()).step_by(width).enumerate() {
                let position = position.checked_add(row % sequence).ok_or(Error::Bounds)?;
                let position = u32::try_from(position).map_err(|_| Error::Bounds)?;
                // Re-census retained templates before every fill, not only the
                // first row. Full caller array backing remains independently
                // charged; borrowed slice elements never authorize its refund.
                let temporary =
                    Self::template_call_temporary(input, &programs, ROTARY_COEFFICIENT_BYTES)?;
                self.native_construction_preflight(
                    tensor::Construction::RotaryHalf {
                        rows: 1,
                        width,
                        cosine_len: half,
                        sine_len: half,
                    },
                    temporary,
                )?;
                fill(
                    theta,
                    position,
                    width,
                    &mut cosine[..half],
                    &mut sine[..half],
                )?;
                let program = tensor::Program::rotary_half(
                    1,
                    width,
                    self.bounds(&input.slice(offset, width)?)?,
                    &cosine[..half],
                    &sine[..half],
                )?;
                programs.push(self.intern_with_temporary(program, temporary)?);
            }
            // No unverified energy is introduced. A future producer-proven complete
            // row certificate can be carried through this exact owner/offset key.
            // Both arrays remain caller-owned through tensor/certificate
            // publication, including the template-list boxing overlap.
            self.tensors(input, width, programs, vec![], ROTARY_COEFFICIENT_BYTES)
        })();
        if result.is_err() {
            self.proof_admission_retired = true;
        }
        result
    }
    pub fn softmax(
        &mut self,
        input: &Layout,
        sequence: usize,
        position: usize,
    ) -> Result<Layout, Error> {
        let keys = position.checked_add(sequence).ok_or(Error::Bounds)?;
        if sequence == 0
            || keys == 0
            || input.len() % keys != 0
            || input.len() / keys % sequence != 0
        {
            return Err(Error::Shape);
        }
        let result = (|| {
            let mut programs = Vec::new();
            let mut certificates = Vec::new();
            self.proof_producer_buffers(input, keys, &mut programs, &mut certificates, 0)?;
            for row in 0..input.len() / keys {
                let width = position + row % sequence + 1;
                self.native_construction_preflight(
                    tensor::Construction::SoftmaxCertified { rows: 1, width },
                    Self::proof_producer_temporary(input, &programs, &certificates, 0)?,
                )?;
                let view = input.slice(row * keys, width)?;
                let (program, mass) =
                    tensor::Program::softmax_certified(1, width, self.bounds(&view)?)?;
                let temporary = Self::proof_producer_temporary(input, &programs, &certificates, 0)?;
                programs.push(self.intern_with_temporary(program, temporary)?);
                certificates.push((row, None, Some(mass)));
            }
            self.tensors(input, keys, programs, certificates, 0)
        })();
        if result.is_err() {
            self.proof_admission_retired = true;
        }
        result
    }
    fn matrix(
        &mut self,
        mut rows: Vec<MatrixRow>,
        lanes: usize,
        rescale: bool,
    ) -> Result<Layout, Error> {
        let mut tiles = 0usize;
        // Validate/price the full public schedule before any region/bounds Vec.
        for row in &mut rows {
            if row.inner == 0
                || row.inner > puma::MAX_LANES
                || row.columns == 0
                || row.left.len() != row.inner
                || row.right.len() != row.inner.checked_mul(row.columns).ok_or(Error::Bounds)?
            {
                return Err(Error::Shape);
            }
            row.tile = (puma::MAX_LANES / row.inner).min(TILE);
            tiles = tiles
                .checked_add(row.columns.div_ceil(row.tile))
                .ok_or(Error::Bounds)?;
        }
        let construction = Self::matrix_construction_bytes(&rows, tiles)?;
        let mut regions = self.region_buffer(tiles, construction)?;
        let mut work = WorkBudget::default();
        let mut advances = 0u64;
        for row in &mut rows {
            let a = self.bounds(&row.left)?;
            let mass = self.proof(&row.left).and_then(|p| p.mass);
            let mut all = Vec::new();
            let count = row.columns.div_ceil(row.tile);
            if all.try_reserve_exact(count).is_err() {
                self.proof_admission_retired = true;
                return Err(Error::Allocation);
            }
            if all.capacity() != count {
                self.proof_admission_retired = true;
                return Err(Error::Bounds);
            }
            for start in (0..row.columns).step_by(row.tile) {
                let n = row.tile.min(row.columns - start);
                let (mut lo, mut hi) = (i64::MAX, i64::MIN);
                for k in 0..row.inner {
                    for c in start..start + n {
                        let range = self.bounds(&row.right.slice(row.right_index(k, c), 1)?)?;
                        lo = lo.min(range.lower());
                        hi = hi.max(range.upper());
                    }
                }
                let bounds = InputBounds {
                    left: a,
                    right: Interval::new(lo, hi)?,
                };
                let mut output = if rescale {
                    bounds.dot_output(row.inner)?
                } else {
                    bounds.dot_product(row.inner)?
                };
                if let Some(mass) = mass {
                    let bound = |v: i64| {
                        i64::try_from(i128::from(v) * i128::from(mass)).map_err(|_| Error::Bounds)
                    };
                    let mut tight = Interval::new(bound(lo.min(0))?, bound(hi.max(0))?)?;
                    if rescale {
                        tight = Interval::new(
                            tight.lower().div_euclid(ONE),
                            -(-tight.upper()).div_euclid(ONE),
                        )?;
                    }
                    output = Interval::new(
                        output.lower().max(tight.lower()),
                        output.upper().min(tight.upper()),
                    )?;
                }
                regions.push(Region {
                    offset: row.output + start,
                    len: n,
                    bounds: output,
                });
                all.push(bounds);
                work = work.checked_add(WorkBudget::product(n, rescale)?)?;
                advances += 1;
            }
            row.bounds = all.into_boxed_slice();
        }
        self.push(
            Op::Matrix {
                rows: rows.into_boxed_slice(),
                rescale,
            },
            lanes,
            regions,
            work,
            advances,
        )
    }
    pub fn dense_private(
        &mut self,
        left: &Layout,
        right: &Layout,
        rows: usize,
        inner: usize,
        columns: usize,
        rescale: bool,
    ) -> Result<Layout, Error> {
        if rows == 0
            || rows > MAX_NODES
            || rows.checked_mul(inner) != Some(left.len())
            || inner.checked_mul(columns) != Some(right.len())
        {
            return Err(Error::Shape);
        }
        let mut schedule = self.matrix_row_buffer(left, right, rows)?;
        for row in 0..rows {
            schedule.push(MatrixRow {
                left: left.slice(row * inner, inner)?,
                right: right.clone(),
                inner,
                columns,
                column_major: true,
                output: row * columns,
                tile: 0,
                bounds: Box::new([]),
            });
        }
        self.matrix(
            schedule,
            rows.checked_mul(columns).ok_or(Error::Bounds)?,
            rescale,
        )
    }
    // Keep the original independently checked public matrix dimensions explicit.
    #[allow(clippy::too_many_arguments)]
    pub fn scores(
        &mut self,
        q: &Layout,
        k: &Layout,
        heads: usize,
        kv_heads: usize,
        sequence: usize,
        position: usize,
        width: usize,
    ) -> Result<Layout, Error> {
        let keys = position.checked_add(sequence).ok_or(Error::Bounds)?;
        if heads == 0
            || kv_heads == 0
            || heads % kv_heads != 0
            || sequence == 0
            || heads.checked_mul(sequence).is_none_or(|n| n > MAX_NODES)
            || heads
                .checked_mul(sequence)
                .and_then(|n| n.checked_mul(width))
                != Some(q.len())
            || kv_heads
                .checked_mul(keys)
                .and_then(|n| n.checked_mul(width))
                != Some(k.len())
        {
            return Err(Error::Shape);
        }
        let mut rows = self.matrix_row_buffer(q, k, heads * sequence)?;
        for head in 0..heads {
            for row in 0..sequence {
                let valid = position + row + 1;
                rows.push(MatrixRow {
                    left: q.slice((head * sequence + row) * width, width)?,
                    right: k.slice((head / (heads / kv_heads)) * keys * width, valid * width)?,
                    inner: width,
                    columns: valid,
                    column_major: true,
                    output: (head * sequence + row) * keys,
                    tile: 0,
                    bounds: Box::new([]),
                });
            }
        }
        self.matrix(
            rows,
            heads
                .checked_mul(sequence)
                .and_then(|n| n.checked_mul(keys))
                .ok_or(Error::Bounds)?,
            true,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn values(
        &mut self,
        p: &Layout,
        v: &Layout,
        heads: usize,
        kv_heads: usize,
        sequence: usize,
        position: usize,
        width: usize,
    ) -> Result<Layout, Error> {
        let keys = position.checked_add(sequence).ok_or(Error::Bounds)?;
        if heads == 0
            || kv_heads == 0
            || heads % kv_heads != 0
            || sequence == 0
            || heads.checked_mul(sequence).is_none_or(|n| n > MAX_NODES)
            || heads
                .checked_mul(sequence)
                .and_then(|n| n.checked_mul(keys))
                != Some(p.len())
            || kv_heads
                .checked_mul(keys)
                .and_then(|n| n.checked_mul(width))
                != Some(v.len())
        {
            return Err(Error::Shape);
        }
        let mut rows = self.matrix_row_buffer(p, v, heads * sequence)?;
        for head in 0..heads {
            for row in 0..sequence {
                let valid = position + row + 1;
                rows.push(MatrixRow {
                    left: p.slice((head * sequence + row) * keys, valid)?,
                    right: v.slice((head / (heads / kv_heads)) * keys * width, valid * width)?,
                    inner: valid,
                    columns: width,
                    column_major: false,
                    output: (head * sequence + row) * width,
                    tile: 0,
                    bounds: Box::new([]),
                });
            }
        }
        self.matrix(
            rows,
            heads
                .checked_mul(sequence)
                .and_then(|n| n.checked_mul(width))
                .ok_or(Error::Bounds)?,
            true,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn public_matrix(
        &mut self,
        input: &Layout,
        source: Arc<Source>,
        rows: usize,
        inner: usize,
        columns: usize,
        transposed: bool,
        rescale: bool,
    ) -> Result<Layout, Error> {
        if rows == 0
            || inner == 0
            || columns == 0
            || rows.checked_mul(inner) != Some(input.len())
            || inner.checked_mul(columns) != Some(source.lanes)
            || (rows as u64)
                .checked_mul(inner as u64)
                .and_then(|n| n.checked_mul(columns as u64))
                .is_none_or(|n| n > 1 << 32)
        {
            return Err(Error::Shape);
        }
        let product = InputBounds {
            left: self.bounds(input)?,
            right: source.bounds,
        }
        .product()?;
        let bounds = multiply::arithmetic::Affine {
            left: i64::try_from(inner).map_err(|_| Error::Bounds)?,
            right: 0,
            constant: 0,
        }
        .output_bounds(InputBounds {
            left: product,
            right: Interval::new(0, 0)?,
        })?;
        let lanes = rows.checked_mul(columns).ok_or(Error::Bounds)?;
        let source_bytes = size_of::<Source>()
            .checked_add(source.key.capacity())
            .and_then(|n| n.checked_add(192))
            .ok_or(Error::Bounds)?;
        let mut regions = self.region_buffer(
            1,
            Self::proof_row_scratch(input)?
                .checked_add(source_bytes)
                .ok_or(Error::Bounds)?,
        )?;
        regions.push(Region {
            offset: 0,
            len: lanes,
            bounds,
        });
        let raw = self.push(
            Op::PublicMatrix {
                input: input.clone(),
                source,
                rows,
                inner,
                columns,
                transposed,
            },
            lanes,
            regions,
            WorkBudget::default(),
            0,
        )?;
        if rescale {
            self.rescale(&raw)
        } else {
            Ok(raw)
        }
    }
    pub fn bias(
        &mut self,
        input: &Layout,
        source: Arc<Source>,
        width: usize,
    ) -> Result<Layout, Error> {
        if width == 0 || width > 8192 || input.len() % width != 0 || source.lanes != width {
            return Err(Error::Shape);
        }
        let source_bytes = size_of::<Source>()
            .checked_add(source.key.capacity())
            .and_then(|n| n.checked_add(192))
            .ok_or(Error::Bounds)?;
        let mut regions = self.region_buffer(
            input.len().div_ceil(TILE),
            Self::proof_row_scratch(input)?
                .checked_add(source_bytes)
                .ok_or(Error::Bounds)?,
        )?;
        for offset in (0..input.len()).step_by(TILE) {
            let len = TILE.min(input.len() - offset);
            let bounds = multiply::arithmetic::Affine {
                left: 1,
                right: 1,
                constant: 0,
            }
            .output_bounds(InputBounds {
                left: self.bounds(&input.slice(offset, len)?)?,
                right: source.bounds,
            })?;
            regions.push(Region {
                offset,
                len,
                bounds,
            });
        }
        self.push(
            Op::Bias {
                input: input.clone(),
                source,
                width,
            },
            input.len(),
            regions,
            WorkBudget::default(),
            0,
        )
    }
}

#[cfg(test)]
mod early_admission_tests {
    use super::*;
    #[test]
    fn tensor_certificate_failure_after_push_preserves_binding_error_and_burns() {
        let mut b = Builder::new(
            WeightMode::Public,
            &[Register {
                lanes: 3,
                bounds: Interval::new(-1, 1).unwrap(),
            }],
        )
        .unwrap();
        let input = b.input(0).unwrap();
        let mut native = tensor::Builder::new(3, Interval::new(-1, 1).unwrap()).unwrap();
        let output = native
            .affine(
                native.input(),
                None,
                multiply::arithmetic::Affine {
                    left: 1,
                    right: 0,
                    constant: 0,
                },
            )
            .unwrap();
        let program = b.intern(native.finish(output).unwrap()).unwrap();
        let work = program.work_budget();
        let advances = 1 + program.instructions() as u64;
        // Invalid private-owner fixture, not a new public certificate API.
        assert!(matches!(
            b.tensors(&input, 3, vec![program], vec![(1, Some(3), None)], 0),
            Err(Error::Binding)
        ));
        assert_eq!(b.ops.len(), 1);
        assert_eq!(b.work, work);
        assert_eq!(b.advances, advances);
        assert!(b.proofs.is_empty());
        assert!(b.proof_admission_retired);
        assert!(matches!(b.input(0), Err(Error::Binding)));
        assert!(matches!(b.finish(&[input]), Err(Error::Binding)));
    }

    #[test]
    fn rescale_factory_failure_after_published_owner_burns() {
        let mut b = Builder::new(
            WeightMode::Public,
            &[Register {
                lanes: 3,
                bounds: Interval::new(-1, 1).unwrap(),
            }],
        )
        .unwrap();
        let input = b.input(0).unwrap();
        let raw = b.binary(&input, &input, false).unwrap();
        let work = b.work;
        let advances = b.advances;
        let called = std::cell::Cell::new(false);
        // Same private chunks boundary used after PublicMatrix pushes raw.
        let result = b.chunks(
            &raw,
            |lanes| tensor::Construction::Builder { lanes },
            |_, _| {
                called.set(true);
                Err(Error::Bounds)
            },
        );
        assert!(matches!(result, Err(Error::Bounds)));
        assert!(called.get());
        assert_eq!(b.ops.len(), 1);
        assert_eq!(b.work, work);
        assert_eq!(b.advances, advances);
        assert!(b.proof_admission_retired);
        assert!(matches!(b.input(0), Err(Error::Binding)));
        assert!(matches!(b.finish(&[raw]), Err(Error::Binding)));
    }

    #[test]
    fn exhausted_body_metadata_rejects_before_native_factory() {
        let mut b = Builder::new(
            WeightMode::Public,
            &[Register {
                lanes: 3,
                bounds: Interval::new(-1, 1).unwrap(),
            }],
        )
        .unwrap();
        let input = b.input(0).unwrap();
        b.proof_metadata_limit = 1;
        let called = std::cell::Cell::new(false);
        let result = b.chunks(
            &input,
            |lanes| tensor::Construction::Builder { lanes },
            |n, _| {
                called.set(true);
                tensor::Program::all_valid(n)
            },
        );
        assert!(matches!(result, Err(Error::Bounds)));
        assert!(!called.get());
        assert!(b.templates.is_empty());
        assert!(b.ops.is_empty());
        assert!(b.proof_admission_retired);
    }
}

#[cfg(test)]
#[path = "body_native_construction_tests.rs"]
mod native_construction_tests;
