//! Complete ring64 products over immutable public i64 little-endian coefficients.
//!
//! Extracted from the replicated decoder's public-source loops. One bounded
//! public scan is shared by one or two components; no resident matrix, float
//! conversion, bias, truncation or requantization is introduced. Callers own
//! source authentication, input/output allocation and any later protected
//! rescale. In particular, a disk chunk is never a rescale boundary.
//!
//! Geometry and every component length are checked before any output mutation.
//! Bounds failures leave outputs untouched. A read failure zeroizes all output
//! components, including any prior accumulator contents. On any error, callers
//! must retire the enclosing operation and must not publish its outputs.
use crate::public_file::PublicFile;
use zeroize::Zeroize;

pub const READ_BUFFER_BYTES: usize = 64 << 10;
pub const MAX_COMPONENTS: usize = 2;
/// Work bound per component, identical to the original public-source loops.
pub const MAX_MULTIPLY_ACCUMULATES: u64 = 1 << 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Bounds,
    Read,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Bounds => "streamed matrix geometry or component bounds differ",
            Self::Read => "streamed matrix public file read failed",
        })
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixLayout {
    /// Coefficient storage is [columns, inner].
    OutputInput,
    /// Coefficient storage is [inner, columns], including Conv1D orientation.
    InputOutput,
}

/// Checked nonempty [rows, inner] x public matrix -> [rows, columns] geometry.
/// The file's byte length and all component lengths are checked at execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixGeometry {
    rows: usize,
    inner: usize,
    columns: usize,
    layout: MatrixLayout,
    input_elements: usize,
    output_elements: usize,
    file_bytes: u64,
}

impl MatrixGeometry {
    pub fn new(
        rows: usize,
        inner: usize,
        columns: usize,
        layout: MatrixLayout,
    ) -> Result<Self, Error> {
        let input_elements = rows.checked_mul(inner).ok_or(Error::Bounds)?;
        let output_elements = rows.checked_mul(columns).ok_or(Error::Bounds)?;
        let file_bytes = coefficient_bytes(inner, columns)?;
        let work = u64::try_from(input_elements)
            .ok()
            .and_then(|n| n.checked_mul(u64::try_from(columns).ok()?))
            .ok_or(Error::Bounds)?;
        if rows == 0 || inner == 0 || columns == 0 || work > MAX_MULTIPLY_ACCUMULATES {
            return Err(Error::Bounds);
        }
        Ok(Self {
            rows,
            inner,
            columns,
            layout,
            input_elements,
            output_elements,
            file_bytes,
        })
    }

    pub fn input_elements(self) -> usize {
        self.input_elements
    }

    pub fn output_elements(self) -> usize {
        self.output_elements
    }

    pub fn coefficient_elements(self) -> usize {
        (self.file_bytes / 8) as usize
    }

    fn coefficient_coordinates(self, index: usize) -> (usize, usize) {
        match self.layout {
            MatrixLayout::InputOutput => (index / self.columns, index % self.columns),
            MatrixLayout::OutputInput => (index % self.inner, index / self.inner),
        }
    }

    // Shared full-ring MAC traversal for public streamed and private resident
    // sources. No allocation, transpose, rounding, or intermediate product.
    fn accumulate_coefficient(
        self,
        index: usize,
        weight: u64,
        inputs: &[&[u64]],
        outputs: &mut [&mut [u64]],
    ) {
        let (k, c) = self.coefficient_coordinates(index);
        for row in 0..self.rows {
            let x = row * self.inner + k;
            let y = row * self.columns + c;
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                output[y] = output[y].wrapping_add(input[x].wrapping_mul(weight));
            }
        }
    }

    /// Optional preflight before a caller allocates outputs. `multiply` repeats
    /// this check, then validates every output before zeroing or accumulating.
    pub fn validate_inputs(self, file: &PublicFile, inputs: &[&[u64]]) -> Result<(), Error> {
        check_inputs(file, self.file_bytes, inputs, self.input_elements)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentError {
    Bounds,
    Cancelled,
}

/// Add complete ring64 resident contraction to caller-owned accumulators.
/// Private coefficients never enter `PublicFile`. All lengths checked before
/// mutation. Cancellation checked before each coefficient (at most `rows` MACs
/// per component between checks), erasing every output on cancellation.
pub fn accumulate_resident(
    coefficients: &[u64],
    inputs: &[&[u64]],
    outputs: &mut [&mut [u64]],
    geometry: MatrixGeometry,
    cancelled: &mut impl FnMut() -> bool,
) -> Result<(), ResidentError> {
    if coefficients.len() != geometry.coefficient_elements()
        || !(1..=MAX_COMPONENTS).contains(&inputs.len())
        || inputs.iter().any(|x| x.len() != geometry.input_elements)
        || check_outputs(inputs, outputs, geometry.output_elements).is_err()
    {
        return Err(ResidentError::Bounds);
    }
    for (index, &weight) in coefficients.iter().enumerate() {
        if cancelled() {
            for output in outputs.iter_mut() {
                output.zeroize();
            }
            return Err(ResidentError::Cancelled);
        }
        geometry.accumulate_coefficient(index, weight, inputs, outputs);
    }
    Ok(())
}

/// Checked public contiguous row selection. Selectors are [rows, count], the
/// complete file is [vocabulary, width], and outputs are [rows, width]. Selection
/// start/count are public; selector values never control reads or iteration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectionGeometry {
    matrix: MatrixGeometry,
    file_bytes: u64,
    start: u64,
    end: u64,
}

impl SelectionGeometry {
    pub fn new(
        rows: usize,
        vocabulary: usize,
        width: usize,
        start: usize,
        count: usize,
    ) -> Result<Self, Error> {
        let end = start.checked_add(count).ok_or(Error::Bounds)?;
        if end > vocabulary {
            return Err(Error::Bounds);
        }
        Ok(Self {
            matrix: MatrixGeometry::new(rows, count, width, MatrixLayout::InputOutput)?,
            file_bytes: coefficient_bytes(vocabulary, width)?,
            start: coefficient_bytes(start, width)?,
            end: coefficient_bytes(end, width)?,
        })
    }
}

fn coefficient_bytes(rows: usize, columns: usize) -> Result<u64, Error> {
    rows.checked_mul(columns)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(Error::Bounds)
}

fn check_inputs(
    file: &PublicFile,
    file_bytes: u64,
    inputs: &[&[u64]],
    elements: usize,
) -> Result<(), Error> {
    if file.size() != file_bytes
        || !(1..=MAX_COMPONENTS).contains(&inputs.len())
        || inputs.iter().any(|input| input.len() != elements)
    {
        return Err(Error::Bounds);
    }
    Ok(())
}

fn check_outputs(inputs: &[&[u64]], outputs: &[&mut [u64]], elements: usize) -> Result<(), Error> {
    if outputs.len() != inputs.len() || outputs.iter().any(|output| output.len() != elements) {
        return Err(Error::Bounds);
    }
    Ok(())
}

/// Replace each output component with its complete ring64 matrix product.
pub fn multiply(
    file: &PublicFile,
    inputs: &[&[u64]],
    outputs: &mut [&mut [u64]],
    geometry: MatrixGeometry,
) -> Result<(), Error> {
    geometry.validate_inputs(file, inputs)?;
    check_outputs(inputs, outputs, geometry.output_elements)?;
    for output in outputs.iter_mut() {
        output.fill(0);
    }
    accumulate(file, inputs, outputs, geometry, 0, geometry.file_bytes)
}

/// Add a public contiguous row range to complete ring64 accumulators. Existing
/// outputs are retained on success, allowing successive public selector tiles.
pub fn accumulate_rows(
    file: &PublicFile,
    selectors: &[&[u64]],
    outputs: &mut [&mut [u64]],
    geometry: SelectionGeometry,
) -> Result<(), Error> {
    check_inputs(
        file,
        geometry.file_bytes,
        selectors,
        geometry.matrix.input_elements,
    )?;
    check_outputs(selectors, outputs, geometry.matrix.output_elements)?;
    accumulate(
        file,
        selectors,
        outputs,
        geometry.matrix,
        geometry.start,
        geometry.end,
    )
}

// Both original loops use the same source-order multiply/add traversal. A row
// selection is an input/output matrix at a public byte offset within the file.
fn accumulate(
    file: &PublicFile,
    inputs: &[&[u64]],
    outputs: &mut [&mut [u64]],
    geometry: MatrixGeometry,
    start: u64,
    end: u64,
) -> Result<(), Error> {
    let mut buffer = [0u8; READ_BUFFER_BYTES];
    let mut offset = start;
    while offset < end {
        let n = buffer.len().min((end - offset) as usize);
        if file.read(offset, &mut buffer[..n]).is_err() {
            for output in outputs.iter_mut() {
                output.zeroize();
            }
            return Err(Error::Read);
        }
        for (i, raw) in buffer[..n].chunks_exact(8).enumerate() {
            let index = ((offset - start) / 8) as usize + i;
            let weight = u64::from_le_bytes(raw.try_into().unwrap());
            geometry.accumulate_coefficient(index, weight, inputs, outputs);
        }
        offset += n as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public_file::PublicFileWriter;
    use num_bigint::BigInt;
    use sha2::{Digest, Sha256};

    fn file(values: &[i64]) -> PublicFile {
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let digest = Sha256::digest(&bytes).into();
        let mut writer = PublicFileWriter::new(bytes.len() as u64, digest).unwrap();
        for chunk in bytes.chunks(READ_BUFFER_BYTES) {
            writer.write(chunk).unwrap();
        }
        let file = writer.finish().unwrap();
        assert_eq!(file.digest(), digest);
        file
    }

    fn component(n: usize, component: usize) -> Vec<u64> {
        (0..n)
            .map(|i| [0, 1, u64::MAX, 1 << 63, (1 << 63) - 1, 4095, 262143][(i + component) % 7])
            .collect()
    }

    fn coefficients(n: usize) -> Vec<i64> {
        (0..n)
            .map(|i| [i64::MIN, i64::MAX, -1, 7, -4097, 262145, 0][i % 7])
            .collect()
    }

    fn ring(sum: BigInt) -> u64 {
        let modulus = BigInt::from(1u8) << 64;
        let reduced: BigInt = ((sum % &modulus) + &modulus) % &modulus;
        reduced.to_u64_digits().1.first().copied().unwrap_or(0)
    }

    // Independent arbitrary-precision integer oracle, output-major traversal.
    // It never calls a production modular arithmetic or geometry helper.
    fn reference(
        input: &[u64],
        weights_oi: &[i64],
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Vec<u64> {
        (0..rows * columns)
            .map(|index| {
                let (row, column) = (index / columns, index % columns);
                let sum = (0..inner)
                    .map(|k| {
                        BigInt::from(input[row * inner + k])
                            * BigInt::from(weights_oi[column * inner + k])
                    })
                    .sum();
                ring(sum)
            })
            .collect()
    }

    #[test]
    fn resident_private_coefficients_share_complete_ring_traversal_and_oracle() {
        let (rows, inner, columns) = (2, 5, 3);
        let weights = coefficients(inner * columns);
        let inputs = [component(rows * inner, 0), component(rows * inner, 1)];
        for layout in [MatrixLayout::OutputInput, MatrixLayout::InputOutput] {
            let stored: Vec<u64> = (0..inner * columns)
                .map(|i| match layout {
                    MatrixLayout::OutputInput => weights[i] as u64,
                    MatrixLayout::InputOutput => {
                        weights[(i % columns) * inner + i / columns] as u64
                    }
                })
                .collect();
            let geometry = MatrixGeometry::new(rows, inner, columns, layout).unwrap();
            let mut outputs = [vec![19; rows * columns], vec![23; rows * columns]];
            let [left, right] = &mut outputs;
            accumulate_resident(
                &stored,
                &[&inputs[0], &inputs[1]],
                &mut [left, right],
                geometry,
                &mut || false,
            )
            .unwrap();
            for (index, (input, output)) in inputs.iter().zip(&outputs).enumerate() {
                let expected: Vec<_> = reference(input, &weights, rows, inner, columns)
                    .into_iter()
                    .map(|v| v.wrapping_add([19, 23][index]))
                    .collect();
                assert_eq!(*output, expected);
            }
        }
    }

    #[test]
    fn resident_bounds_leave_all_outputs_unchanged_cancel_erases_all() {
        let geometry = MatrixGeometry::new(2, 3, 2, MatrixLayout::InputOutput).unwrap();
        for (weights, inputs, mut outputs) in [
            (vec![1; 5], vec![vec![1; 6]], vec![vec![9; 4]]),
            (vec![1; 6], vec![vec![1; 5]], vec![vec![9; 4]]),
            (
                vec![1; 6],
                vec![vec![1; 6]; 2],
                vec![vec![9; 4], vec![9; 3]],
            ),
            (vec![1; 6], vec![vec![1; 6]; 3], vec![vec![9; 4]; 3]),
            (vec![1; 6], vec![], vec![]),
        ] {
            let original = outputs.clone();
            let mut checks = 0;
            assert_eq!(
                accumulate_resident(
                    &weights,
                    &inputs.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &mut outputs
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>(),
                    geometry,
                    &mut || {
                        checks += 1;
                        false
                    }
                ),
                Err(ResidentError::Bounds)
            );
            assert_eq!(outputs, original);
            assert_eq!(checks, 0);
        }
        for fail in 0..6 {
            let mut outputs = [vec![9; 4], vec![9; 4]];
            let [left, right] = &mut outputs;
            let mut checks = 0;
            assert_eq!(
                accumulate_resident(
                    &[1; 6],
                    &[&[1; 6], &[1; 6]],
                    &mut [left, right],
                    geometry,
                    &mut || {
                        checks += 1;
                        checks == fail + 1
                    }
                ),
                Err(ResidentError::Cancelled)
            );
            assert_eq!(outputs, [vec![0; 4], vec![0; 4]]);
        }
    }

    #[test]
    fn one_and_two_components_match_integer_oracle_in_both_storage_orientations() {
        for (rows, inner, columns) in [(2, 5, 3), (1, 1, 1)] {
            let weights = coefficients(inner * columns);
            for layout in [MatrixLayout::OutputInput, MatrixLayout::InputOutput] {
                let stored = match layout {
                    MatrixLayout::OutputInput => weights.clone(),
                    MatrixLayout::InputOutput => (0..inner * columns)
                        .map(|i| weights[(i % columns) * inner + i / columns])
                        .collect(),
                };
                let source = file(&stored);
                let digest = source.digest();
                let geometry = MatrixGeometry::new(rows, inner, columns, layout).unwrap();
                for count in 1..=2 {
                    let inputs: Vec<_> = (0..count).map(|c| component(rows * inner, c)).collect();
                    let mut outputs = vec![vec![19; rows * columns]; count];
                    multiply(
                        &source,
                        &inputs.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                        &mut outputs
                            .iter_mut()
                            .map(Vec::as_mut_slice)
                            .collect::<Vec<_>>(),
                        geometry,
                    )
                    .unwrap();
                    for (input, output) in inputs.iter().zip(&outputs) {
                        assert_eq!(*output, reference(input, &weights, rows, inner, columns));
                    }
                    assert_eq!(source.digest(), digest);
                }
            }
        }
    }

    #[test]
    fn complete_sums_span_short_exact_and_multiple_read_chunks() {
        for (inner, columns) in [(8191, 1), (8192, 1), (8193, 1), (4099, 3)] {
            let rows = 2;
            let weights = coefficients(inner * columns);
            for layout in [MatrixLayout::OutputInput, MatrixLayout::InputOutput] {
                let stored = match layout {
                    MatrixLayout::OutputInput => weights.clone(),
                    MatrixLayout::InputOutput => (0..inner * columns)
                        .map(|i| weights[(i % columns) * inner + i / columns])
                        .collect(),
                };
                let inputs = [component(rows * inner, 0), component(rows * inner, 1)];
                let mut outputs = [vec![0; rows * columns], vec![0; rows * columns]];
                multiply(
                    &file(&stored),
                    &[&inputs[0], &inputs[1]],
                    &mut outputs
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>(),
                    MatrixGeometry::new(rows, inner, columns, layout).unwrap(),
                )
                .unwrap();
                for (input, output) in inputs.iter().zip(&outputs) {
                    assert_eq!(*output, reference(input, &weights, rows, inner, columns));
                }
            }
        }
    }

    #[test]
    fn selectors_add_only_public_ranges_and_preserve_prior_accumulation() {
        for (vocabulary, width, start, count) in [
            (7, 5, 0, 1),
            (7, 5, 2, 3),
            (7, 5, 6, 1),
            (7, 5, 0, 7),
            (5, 4099, 1, 3),
        ] {
            let rows = 2;
            let weights = coefficients(vocabulary * width);
            let weights_oi: Vec<_> = (0..width * count)
                .map(|i| weights[(start + i % count) * width + i / count])
                .collect();
            for components in 1..=2 {
                let inputs: Vec<_> = (0..components)
                    .map(|c| component(rows * count, c))
                    .collect();
                let mut outputs: Vec<_> = (0..components)
                    .map(|c| component(rows * width, c))
                    .collect();
                let initial = outputs.clone();
                accumulate_rows(
                    &file(&weights),
                    &inputs.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &mut outputs
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>(),
                    SelectionGeometry::new(rows, vocabulary, width, start, count).unwrap(),
                )
                .unwrap();
                for c in 0..components {
                    let sums = reference(&inputs[c], &weights_oi, rows, count, width);
                    let expected: Vec<_> = initial[c]
                        .iter()
                        .zip(sums)
                        .map(|(&prior, sum)| ring(BigInt::from(prior) + BigInt::from(sum)))
                        .collect();
                    assert_eq!(outputs[c], expected);
                }
            }
        }
    }

    #[test]
    fn invalid_dimensions_ranges_overflows_and_work_reject_without_allocation() {
        for (rows, inner, columns) in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (usize::MAX, 2, 1),
            (1, usize::MAX, 2),
            (2, 1, usize::MAX),
            (65536, 65536, 2),
        ] {
            assert_eq!(
                MatrixGeometry::new(rows, inner, columns, MatrixLayout::OutputInput),
                Err(Error::Bounds)
            );
        }
        let exact = MatrixGeometry::new(65536, 65536, 1, MatrixLayout::InputOutput).unwrap();
        assert_eq!(exact.input_elements() as u64, MAX_MULTIPLY_ACCUMULATES);
        assert_eq!(exact.output_elements(), 65536);
        for (rows, vocabulary, width, start, count) in [
            (0, 2, 1, 0, 1),
            (1, 0, 1, 0, 1),
            (1, 2, 0, 0, 1),
            (1, 2, 1, 0, 0),
            (1, 2, 1, 2, 1),
            (1, 2, 1, 1, 2),
            (1, 2, 1, usize::MAX, 2),
            (1, usize::MAX, 2, 0, 1),
            (65536, 65536, 2, 0, 65536),
        ] {
            assert_eq!(
                SelectionGeometry::new(rows, vocabulary, width, start, count),
                Err(Error::Bounds)
            );
        }
        assert!(SelectionGeometry::new(65536, 65536, 1, 0, 65536).is_ok());
    }

    #[test]
    fn every_component_and_file_size_is_checked_before_mutating_any_output() {
        let source = file(&[1; 6]);
        let short_source = file(&[1; 5]);
        let matrix = MatrixGeometry::new(2, 3, 2, MatrixLayout::InputOutput).unwrap();
        let selection = SelectionGeometry::new(2, 3, 2, 0, 3).unwrap();
        for (source, inputs, mut outputs) in [
            (&source, vec![], vec![vec![9; 4]]),
            (&source, vec![vec![1; 6]; 3], vec![vec![9; 4]; 3]),
            (&source, vec![vec![1; 5]], vec![vec![9; 4]]),
            (&source, vec![vec![1; 6], vec![1; 5]], vec![vec![9; 4]; 2]),
            (&source, vec![vec![1; 6]], vec![]),
            (&source, vec![vec![1; 6]; 2], vec![vec![9; 4]]),
            (&source, vec![vec![1; 6]], vec![vec![9; 4]; 2]),
            (&source, vec![vec![1; 6]; 2], vec![vec![9; 4], vec![9; 3]]),
            (&source, vec![vec![1; 6]], vec![vec![9; 5]]),
            (&short_source, vec![vec![1; 6]; 2], vec![vec![9; 4]; 2]),
        ] {
            let original = outputs.clone();
            let inputs: Vec<_> = inputs.iter().map(Vec::as_slice).collect();
            assert_eq!(
                multiply(
                    source,
                    &inputs,
                    &mut outputs
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>(),
                    matrix
                ),
                Err(Error::Bounds)
            );
            assert_eq!(outputs, original);
            assert_eq!(
                accumulate_rows(
                    source,
                    &inputs,
                    &mut outputs
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>(),
                    selection
                ),
                Err(Error::Bounds)
            );
            assert_eq!(outputs, original);
        }
    }

    #[test]
    fn read_failure_after_partial_accumulation_zeroizes_all_components() {
        let count = READ_BUFFER_BYTES / 8 + 1;
        let source = file(&vec![1; count]);
        // Test-only access to the internal descriptor models a failed second
        // read. Production callers cannot mutate a completed PublicFile.
        source.file().set_len(READ_BUFFER_BYTES as u64).unwrap();
        let input = vec![1; count];
        let mut first = [11];
        let mut second = [22];
        assert_eq!(
            multiply(
                &source,
                &[&input, &input],
                &mut [&mut first, &mut second],
                MatrixGeometry::new(1, count, 1, MatrixLayout::OutputInput).unwrap()
            ),
            Err(Error::Read)
        );
        assert_eq!((first, second), ([0], [0]));
        first[0] = 11;
        second[0] = 22;
        assert_eq!(
            accumulate_rows(
                &source,
                &[&input, &input],
                &mut [&mut first, &mut second],
                SelectionGeometry::new(1, count, 1, 0, count).unwrap()
            ),
            Err(Error::Read)
        );
        assert_eq!((first, second), ([0], [0]));
    }
}
