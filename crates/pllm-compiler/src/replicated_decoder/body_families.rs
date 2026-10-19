//! Original constructors and public loop geometry for every layer family.
use super::*;

impl Builder {
    pub fn binary(
        &mut self,
        left: &Layout,
        right: &Layout,
        multiply: bool,
    ) -> Result<Layout, Error> {
        if left.len() != right.len() {
            return Err(Error::Shape);
        }
        let mut regions = Vec::new();
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
        self.push(
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
        let output = self.binary(left, right, false)?;
        for (row, squared) in energies.into_iter().enumerate() {
            self.proofs.push(Proof {
                key: output
                    .slice(row * width, width)?
                    .row_key(width)
                    .ok_or(Error::Binding)?,
                energy: Some(squared),
                mass: None,
            });
        }
        Ok(output)
    }
    fn tensors(
        &mut self,
        input: &Layout,
        stride: usize,
        programs: Vec<Arc<tensor::Program>>,
        certificates: Vec<(usize, Option<u128>, Option<u64>)>,
    ) -> Result<Layout, Error> {
        if stride == 0 || input.len().div_ceil(stride) != programs.len() {
            return Err(Error::Shape);
        }
        let mut interned = Vec::new();
        let mut regions = Vec::new();
        let mut work = WorkBudget::default();
        let mut advances = 0u64;
        for (row, program) in programs.into_iter().enumerate() {
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
            interned.push(program);
        }
        let output = self.push(
            Op::Tensor {
                input: input.clone(),
                stride,
                programs: interned.into_boxed_slice(),
            },
            input.len(),
            regions,
            work,
            advances,
        )?;
        for (row, energy, mass) in certificates {
            let offset = row * stride;
            let width = match &self.ops[self.ops.len() - 1] {
                Op::Tensor { programs, .. } => programs[row].output_lanes(),
                _ => return Err(Error::Binding),
            };
            self.proofs.push(Proof {
                key: output
                    .slice(offset, width)?
                    .row_key(width)
                    .ok_or(Error::Binding)?,
                energy,
                mass,
            });
        }
        Ok(output)
    }
    fn chunks(
        &mut self,
        input: &Layout,
        make: impl Fn(usize, Interval) -> Result<tensor::Program, Error>,
    ) -> Result<Layout, Error> {
        let mut programs = Vec::new();
        for offset in (0..input.len()).step_by(TILE) {
            let n = TILE.min(input.len() - offset);
            let program = make(n, self.bounds(&input.slice(offset, n)?)?)?;
            programs.push(self.intern(program, 0)?);
        }
        self.tensors(input, TILE, programs, vec![])
    }
    pub fn rescale(&mut self, input: &Layout) -> Result<Layout, Error> {
        self.chunks(input, |n, bounds| {
            let mut b = tensor::Builder::new(n, bounds)?;
            let output = b.rescale(b.input(), 18)?;
            b.finish(output)
        })
    }
    /// Deliberately includes masked slots. Zero values do not elide any rescale.
    pub fn attention_scale(&mut self, input: &Layout, head_dim: usize) -> Result<Layout, Error> {
        if head_dim == 0 {
            return Err(Error::Shape);
        }
        let coefficient = ((1.0_f32 / (head_dim as f32).sqrt()) * ONE as f32) as i64;
        self.chunks(input, |n, bounds| {
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
        })
    }
    pub fn silu(&mut self, input: &Layout) -> Result<Layout, Error> {
        self.chunks(input, |n, bounds| {
            tensor::Program::activation_guarded(tensor::Activation::Silu, n, bounds)
        })
    }
    /// All complete membership tiles and sticky products precede normalization.
    pub fn checked(
        &mut self,
        input: &Layout,
        bound: i64,
        previous: Option<&Layout>,
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
        let mut checked = Vec::new();
        let mut valid = Vec::new();
        let mut work = WorkBudget::default();
        let mut advances = 0u64;
        for (tile, offset) in (0..input.len()).step_by(tensor::MAX_TILE_LANES).enumerate() {
            let n = tensor::MAX_TILE_LANES.min(input.len() - offset);
            let a = self.intern(
                tensor::Program::checked_range(
                    n,
                    self.bounds(&input.slice(offset, n)?)?,
                    accepted,
                    0,
                )?,
                0,
            )?;
            let b = self.intern(tensor::Program::all_valid(n)?, 0)?;
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
        let output = self.push(
            Op::Check {
                input: input.clone(),
                previous: previous.cloned(),
                accepted,
                checked: checked.into_boxed_slice(),
                valid: valid.into_boxed_slice(),
            },
            n.checked_add(1).ok_or(Error::Bounds)?,
            vec![
                Region {
                    offset: 0,
                    len: n,
                    bounds: accepted,
                },
                Region {
                    offset: n,
                    len: 1,
                    bounds: Interval::new(0, 1)?,
                },
            ],
            work,
            advances,
        )?;
        Ok((output.slice(0, n)?, Some(output.slice(n, 1)?)))
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
        let (safe, valid) = match checked {
            Some(bound) => self.checked(input, bound, previous)?,
            None => (input.clone(), previous.cloned()),
        };
        let mut programs = Vec::new();
        let mut certificates = Vec::new();
        for (row, offset) in (0..safe.len()).step_by(width).enumerate() {
            let input = safe.slice(offset, width)?;
            let bounds = self.bounds(&input)?;
            let (program, squared) = match self.proof(&input).and_then(|p| p.energy) {
                Some(norm) => {
                    tensor::Program::rms_norm_with_input_norm(1, width, bounds, gamma, norm)?
                }
                None => tensor::Program::rms_norm_certified(1, width, bounds, gamma)?,
            };
            programs.push(self.intern(program, width)?);
            certificates.push((row, Some(squared), None));
        }
        Ok((self.tensors(&safe, width, programs, certificates)?, valid))
    }
    pub fn rope(
        &mut self,
        input: &Layout,
        width: usize,
        sequence: usize,
        position: usize,
        theta: f32,
    ) -> Result<Layout, Error> {
        if width == 0
            || width > 128
            || width % 2 != 0
            || sequence == 0
            || input.len() % width != 0
            || input.len() / width % sequence != 0
        {
            return Err(Error::Shape);
        }
        let mut programs = Vec::new();
        for (row, offset) in (0..input.len()).step_by(width).enumerate() {
            let (cos, sin) = tensor::Program::rotary_coefficients(
                theta,
                u32::try_from(position + row % sequence).map_err(|_| Error::Bounds)?,
                width,
            )?;
            let program = tensor::Program::rotary_half(
                1,
                width,
                self.bounds(&input.slice(offset, width)?)?,
                &cos,
                &sin,
            )?;
            programs.push(self.intern(program, 2 * width)?);
        }
        // No unverified energy is introduced. A future producer-proven complete
        // row certificate can be carried through this exact owner/offset key.
        self.tensors(input, width, programs, vec![])
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
        let mut programs = Vec::new();
        let mut certificates = Vec::new();
        for row in 0..input.len() / keys {
            let width = position + row % sequence + 1;
            let view = input.slice(row * keys, width)?;
            let (program, mass) =
                tensor::Program::softmax_certified(1, width, self.bounds(&view)?)?;
            programs.push(self.intern(program, 0)?);
            certificates.push((row, None, Some(mass)));
        }
        self.tensors(input, keys, programs, certificates)
    }
    fn matrix(
        &mut self,
        mut rows: Vec<MatrixRow>,
        lanes: usize,
        rescale: bool,
    ) -> Result<Layout, Error> {
        let mut regions = Vec::new();
        let mut work = WorkBudget::default();
        let mut advances = 0u64;
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
            let a = self.bounds(&row.left)?;
            let mass = self.proof(&row.left).and_then(|p| p.mass);
            let mut all = Vec::new();
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
        let mut schedule = Vec::new();
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
        let mut rows = Vec::new();
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
        let mut rows = Vec::new();
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
            vec![Region {
                offset: 0,
                len: lanes,
                bounds,
            }],
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
        let mut regions = Vec::new();
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
