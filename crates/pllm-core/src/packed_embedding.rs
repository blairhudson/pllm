//! Bounded coefficient packing for one-hot embedding queries.
//!
//! This is integer algebra, not encryption. A query must contain exactly one
//! unit coefficient for the no-accumulation width argument to apply.

pub const COEFFICIENT_BITS: usize = 13;
pub const MODULUS: u16 = 1 << COEFFICIENT_BITS;
pub const POLY_DEGREE: usize = 4096;
pub const MAX_TABLE_ELEMENTS: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Int8,
    Int4,
    Mixed433,
}

impl Layout {
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "int8" => Ok(Self::Int8),
            "int4" => Ok(Self::Int4),
            "mixed_4_3_3" => Ok(Self::Mixed433),
            _ => Err("unsupported embedding coefficient layout".into()),
        }
    }

    pub fn widths(self) -> &'static [usize] {
        match self {
            Self::Int8 => &[8],
            Self::Int4 => &[4, 4],
            Self::Mixed433 => &[4, 3, 3],
        }
    }

    pub fn pack(self, values: &[i8]) -> Result<u16, String> {
        if values.is_empty() || values.len() > self.widths().len() {
            return Err("invalid packed embedding channel count".into());
        }
        let mut word = 0_u16;
        let mut offset = 0;
        for (&value, &bits) in values.iter().zip(self.widths()) {
            let limit = 1_i16 << (bits - 1);
            if i16::from(value) >= limit || i16::from(value) < -limit {
                return Err("embedding code exceeds its public channel width".into());
            }
            word |= ((value as u16) & ((1 << bits) - 1)) << offset;
            offset += bits + 1;
        }
        Ok(word)
    }

    pub fn unpack(self, word: u16) -> Result<Vec<i8>, String> {
        let mut offset = 0;
        let mut output = Vec::with_capacity(self.widths().len());
        for &bits in self.widths() {
            if word & (1 << (offset + bits)) != 0 {
                return Err("nonzero embedding guard bit".into());
            }
            let raw = (word >> offset) & ((1 << bits) - 1);
            let signed = if raw & (1 << (bits - 1)) != 0 {
                i16::try_from(raw).expect("bounded code") - (1 << bits)
            } else {
                raw as i16
            };
            output.push(signed as i8);
            offset += bits + 1;
        }
        if word >> offset != 0 {
            return Err("nonzero embedding coefficient padding".into());
        }
        Ok(output)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    rows: usize,
    columns: usize,
    padded_rows: usize,
    packed_columns: usize,
    columns_per_polynomial: usize,
    polynomials: usize,
}

impl Geometry {
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn columns(self) -> usize {
        self.columns
    }
    pub fn padded_rows(self) -> usize {
        self.padded_rows
    }
    pub fn packed_columns(self) -> usize {
        self.packed_columns
    }
    pub fn polynomials(self) -> usize {
        self.polynomials
    }

    pub fn new(rows: usize, columns: usize, layout: Layout) -> Result<Self, String> {
        if rows == 0 || rows > POLY_DEGREE || columns == 0 || columns > 256 {
            return Err("embedding geometry requires 1..4096 rows and 1..256 columns".into());
        }
        if rows
            .checked_mul(columns)
            .is_none_or(|n| n > MAX_TABLE_ELEMENTS)
        {
            return Err("embedding table exceeds the allocation bound".into());
        }
        let padded_rows = rows.next_power_of_two();
        let packed_columns = columns.div_ceil(layout.widths().len());
        let columns_per_polynomial = POLY_DEGREE / padded_rows;
        Ok(Self {
            rows,
            columns,
            padded_rows,
            packed_columns,
            columns_per_polynomial,
            polynomials: packed_columns.div_ceil(columns_per_polynomial),
        })
    }

    pub fn output_coefficient(self, packed_column: usize) -> Result<(usize, usize), String> {
        if packed_column >= self.packed_columns {
            return Err("packed output column exceeds the committed shape".into());
        }
        Ok((
            packed_column / self.columns_per_polynomial,
            (packed_column % self.columns_per_polynomial + 1) * self.padded_rows - 1,
        ))
    }

    pub fn query_coefficient(self, row: usize) -> Result<usize, String> {
        if row >= self.rows {
            return Err("embedding row exceeds the public table".into());
        }
        Ok(self.padded_rows - 1 - row)
    }
}

/// The caller owns the original codes. These independent coefficient buffers
/// are priced separately and never indexed by a private token on the provider.
pub fn pack_table(
    codes: &[i8],
    geometry: Geometry,
    layout: Layout,
) -> Result<Vec<Vec<u16>>, String> {
    if geometry != Geometry::new(geometry.rows, geometry.columns, layout)?
        || codes.len() != geometry.rows * geometry.columns
    {
        return Err("embedding table does not match its public geometry".into());
    }
    let mut polynomials = vec![vec![0_u16; POLY_DEGREE]; geometry.polynomials];
    for row in 0..geometry.rows {
        for (column, channels) in codes[row * geometry.columns..(row + 1) * geometry.columns]
            .chunks(layout.widths().len())
            .enumerate()
        {
            let polynomial = column / geometry.columns_per_polynomial;
            let coefficient =
                (column % geometry.columns_per_polynomial) * geometry.padded_rows + row;
            polynomials[polynomial][coefficient] = layout.pack(channels)?;
        }
    }
    Ok(polynomials)
}

/// Public offline channel statistics, in bounded row chunks. The caller combines
/// sums across the complete source table before freezing any numeric profile.
pub fn channel_statistics(weights: &[f32], columns: usize) -> Result<(Vec<f32>, Vec<f64>), String> {
    if columns == 0
        || columns > 8192
        || weights.is_empty()
        || weights.len() > MAX_TABLE_ELEMENTS
        || weights.len().checked_rem(columns) != Some(0)
        || weights.iter().any(|v| !v.is_finite())
    {
        return Err("invalid bounded embedding statistics input".into());
    }
    let mut maxima = vec![0_f32; columns];
    let mut sums = vec![0_f64; columns];
    for row in weights.chunks_exact(columns) {
        for (i, &value) in row.iter().enumerate() {
            maxima[i] = maxima[i].max(value.abs());
            sums[i] += f64::from(value.abs());
        }
    }
    Ok((maxima, sums))
}

/// Frozen public per-channel power-of-two scales. Symmetric saturation is an
/// explicit quantizer operation, separate from exact signed-code HE execution.
pub fn quantize_channels(
    weights: &[f32],
    widths: &[u8],
    scales: &[f32],
) -> Result<Vec<i8>, String> {
    let columns = widths.len();
    if columns == 0
        || columns > 8192
        || scales.len() != columns
        || weights.is_empty()
        || weights.len() > MAX_TABLE_ELEMENTS
        || weights.len().checked_rem(columns) != Some(0)
        || weights.iter().any(|v| !v.is_finite())
        || widths.iter().any(|w| ![3, 4, 8].contains(w))
        || scales.iter().any(|&s| {
            !s.is_normal() || s <= 0.0 || s > f32::MAX / 127.0 || s.to_bits() & ((1 << 23) - 1) != 0
        })
    {
        return Err("invalid frozen embedding quantizer input".into());
    }
    Ok(weights
        .iter()
        .enumerate()
        .map(|(i, &weight)| {
            let column = i % columns;
            let qmax = f64::from((1_u16 << (widths[column] - 1)) - 1);
            (f64::from(weight) / f64::from(scales[column]))
                .round_ties_even()
                .clamp(-qmax, qmax) as i8
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_codes_and_guard_bits_have_an_exact_contract() {
        for layout in [Layout::Int8, Layout::Int4, Layout::Mixed433] {
            for code in i8::MIN..=i8::MAX {
                let values = layout
                    .widths()
                    .iter()
                    .map(|&b| {
                        let m = 1_i16 << b;
                        (i16::from(code).rem_euclid(m) - m / 2) as i8
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    layout.unpack(layout.pack(&values).unwrap()).unwrap(),
                    values
                );
            }
            assert!(layout.unpack(1 << layout.widths()[0]).is_err());
        }
        assert!(Layout::Mixed433.pack(&[8, 0, 0]).is_err());
        assert!(Layout::Int4.pack(&[0, -9]).is_err());
        assert!(Layout::Int8.unpack(1 << 12).is_err());
    }

    #[test]
    fn negacyclic_one_hot_products_match_every_row_without_clear_index_gathers() {
        for rows in [1, 3, 16, 65, 4096] {
            for layout in [Layout::Int8, Layout::Int4, Layout::Mixed433] {
                let geometry = Geometry::new(rows, 7, layout).unwrap();
                let codes = (0..rows * 7)
                    .map(|i| ((i * 17 + 3) % 7) as i8 - 3)
                    .collect::<Vec<_>>();
                let polynomials = pack_table(&codes, geometry, layout).unwrap();
                for row in [0, rows / 2, rows - 1] {
                    // Independent full negacyclic shift, rather than using the
                    // table-packing offset to fetch the expected coefficient.
                    let exponent = geometry.query_coefficient(row).unwrap();
                    let products = polynomials
                        .iter()
                        .map(|p| {
                            let mut product = vec![0_u16; POLY_DEGREE];
                            for (i, &c) in p.iter().enumerate() {
                                let position = i + exponent;
                                product[position % POLY_DEGREE] = if position >= POLY_DEGREE {
                                    MODULUS.wrapping_sub(c) % MODULUS
                                } else {
                                    c
                                };
                            }
                            product
                        })
                        .collect::<Vec<_>>();
                    let mut result = Vec::new();
                    for column in 0..geometry.packed_columns {
                        let (p, i) = geometry.output_coefficient(column).unwrap();
                        result.extend(layout.unpack(products[p][i]).unwrap());
                    }
                    result.truncate(geometry.columns);
                    assert_eq!(result, codes[row * 7..(row + 1) * 7]);
                }
                assert!(geometry.query_coefficient(rows).is_err());
            }
        }
        assert!(Geometry::new(4097, 1, Layout::Int8).is_err());
        assert!(Geometry::new(1, 257, Layout::Int8).is_err());
    }

    #[test]
    fn sharewise_unpacking_loses_cross_channel_carries() {
        let layout = Layout::Mixed433;
        let packed = layout.pack(&[1, 2, 3]).unwrap();
        let mask = 31_u16;
        let client = packed.wrapping_sub(mask) & (MODULUS - 1);
        let naive_second = (((client >> 5) & 7) + ((mask >> 5) & 7)) & 7;
        assert_ne!(naive_second, 2);
        assert_eq!(
            layout
                .unpack(client.wrapping_add(mask) & (MODULUS - 1))
                .unwrap(),
            [1, 2, 3]
        );
    }

    #[test]
    fn public_quantizer_defines_ties_saturation_and_rejects_invalid_profiles() {
        let weights = [-3.0, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 9.0];
        assert_eq!(
            quantize_channels(&weights, &[3], &[1.0]).unwrap(),
            [-3, -2, -2, 0, 0, 2, 2, 3]
        );
        let (maxima, sums) = channel_statistics(&weights, 2).unwrap();
        assert_eq!(maxima, [3.0, 9.0]);
        assert_eq!(sums, [7.5, 13.5]);
        for scale in [0.0, -1.0, 0.3, f32::NAN, f32::INFINITY, f32::MAX] {
            assert!(quantize_channels(&weights, &[4], &[scale]).is_err());
        }
        assert!(quantize_channels(&weights, &[2], &[1.0]).is_err());
        assert!(channel_statistics(&[f32::NAN], 1).is_err());
        assert!(channel_statistics(&weights, 3).is_err());
    }
}
