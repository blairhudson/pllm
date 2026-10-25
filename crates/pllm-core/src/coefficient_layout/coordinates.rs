//! Shared coordinate owner for legacy layout and the clear ring64 path.
//! Does not construct an odd ring or relax legacy degree/inverse defaults.
use super::LayoutError as Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Coordinate {
    pub matrix: usize,
    pub coefficient: usize,
    pub negative: bool,
}

/// Checked geometry only; not a ring, capability or source certificate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Coordinates {
    degree: usize,
    shape: [usize; 3],
    window: [usize; 3],
    blocks: [usize; 3],
}
impl Coordinates {
    pub fn bounded(
        degree: usize,
        shape: [usize; 3],
        window: [usize; 3],
        max_degree: usize,
    ) -> Result<Self, Error> {
        let [kw, mw, nw] = window;
        if !degree.is_power_of_two()
            || degree > max_degree
            || degree > 8192
            || shape.contains(&0)
            || window.contains(&0)
            || !mw.is_power_of_two()
            || mw > 256
        {
            return Err(Error::InvalidGeometry);
        }
        let volume = kw
            .checked_mul(mw)
            .and_then(|x| x.checked_mul(nw))
            .ok_or(Error::Overflow)?;
        if volume > degree {
            return Err(Error::InvalidGeometry);
        }
        for (a, b) in [
            (shape[0], shape[1]),
            (shape[1], shape[2]),
            (shape[0], shape[2]),
        ] {
            a.checked_mul(b).ok_or(Error::Overflow)?;
        }
        let blocks = [
            shape[0].div_ceil(kw),
            shape[1].div_ceil(mw),
            shape[2].div_ceil(nw),
        ];
        blocks[0].checked_mul(blocks[2]).ok_or(Error::Overflow)?;
        Ok(Self {
            degree,
            shape,
            window,
            blocks,
        })
    }
    /// Public support only. Zero values are NOT skipped.
    pub fn lhs(
        self,
        row_block: usize,
        inner_block: usize,
    ) -> Result<impl Iterator<Item = Coordinate>, Error> {
        if row_block >= self.blocks[0] || inner_block >= self.blocks[1] {
            return Err(Error::InvalidGeometry);
        }
        let [kw, mw, nw] = self.window;
        let start_r = row_block * kw;
        let start_h = inner_block * mw;
        Ok((0..kw.min(self.shape[0] - start_r)).flat_map(move |r| {
            (0..mw.min(self.shape[1] - start_h)).map(move |h| {
                let base = r * nw * mw;
                Coordinate {
                    matrix: (start_r + r) * self.shape[1] + start_h + h,
                    coefficient: if base >= h {
                        base - h
                    } else {
                        self.degree - (h - base)
                    },
                    negative: base < h,
                }
            })
        }))
    }
    pub fn rhs(
        self,
        inner_block: usize,
        col_block: usize,
    ) -> Result<impl Iterator<Item = Coordinate>, Error> {
        if inner_block >= self.blocks[1] || col_block >= self.blocks[2] {
            return Err(Error::Index);
        }
        let [_, mw, nw] = self.window;
        let start_h = inner_block * mw;
        let start_c = col_block * nw;
        Ok((0..mw.min(self.shape[1] - start_h)).flat_map(move |h| {
            (0..nw.min(self.shape[2] - start_c)).map(move |c| Coordinate {
                matrix: (start_h + h) * self.shape[2] + start_c + c,
                coefficient: c * mw + h,
                negative: false,
            })
        }))
    }
    pub fn address(self, row: usize, col: usize) -> Result<(usize, usize), Error> {
        if row >= self.shape[0] || col >= self.shape[2] {
            return Err(Error::Index);
        }
        let [kw, mw, nw] = self.window;
        let t = row / kw * self.blocks[2] + col / nw;
        Ok((t / mw, ((row % kw) * nw + col % nw) * mw + t % mw))
    }
    /// Whether a traced coefficient is a non-padding output cell. All other
    /// coefficients of packed matrix products are structurally zero, regardless
    /// of input values. This inverts address(), not a value-dependent zero test.
    pub fn visible(self, group: usize, coefficient: usize) -> Result<bool, Error> {
        if coefficient >= self.degree {
            return Err(Error::Index);
        }
        let [kw, mw, nw] = self.window;
        let t = group
            .checked_mul(mw)
            .and_then(|v| v.checked_add(coefficient % mw))
            .ok_or(Error::Overflow)?;
        if t >= self.blocks[0] * self.blocks[2] {
            return Ok(false);
        }
        let local = coefficient / mw;
        let row_start = (t / self.blocks[2])
            .checked_mul(kw)
            .ok_or(Error::Overflow)?;
        let col_start = (t % self.blocks[2])
            .checked_mul(nw)
            .ok_or(Error::Overflow)?;
        let rows = self.shape[0]
            .checked_sub(row_start)
            .ok_or(Error::Overflow)?;
        let cols = self.shape[2]
            .checked_sub(col_start)
            .ok_or(Error::Overflow)?;
        let row = local / nw;
        let col = local % nw;
        Ok(row < kw && row < rows && col < cols)
    }
}
