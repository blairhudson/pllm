//! Immutable, file-backed signed-i8 matrices with bounded exact page execution.
//! A verified private anonymous snapshot severs subsequent source-file mutation.
use crate::kernels::{Executor, Matrix};
use flate2::{write::ZlibEncoder, Compression, Decompress, FlushDecompress, Status};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

const MAGIC: &[u8; 8] = b"PLLMWM01";
const HEADER: usize = 64;
const ENTRY: usize = 24;
const MAX_WEIGHT_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_PAGE_BYTES: usize = 4 * 1024 * 1024;
const MAX_PAGES: usize = 65536;
const MAX_ELEMENTS: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy)]
struct Page {
    offset: u64,
    stored: usize,
    decoded: usize,
    compressed: bool,
}

fn shape(rows: usize, cols: usize, page_rows: usize) -> Result<(usize, usize), String> {
    let bytes = rows
        .checked_mul(cols)
        .ok_or("paged matrix shape overflow")?;
    let page_bytes = cols
        .checked_mul(page_rows)
        .ok_or("paged matrix page overflow")?;
    if rows == 0
        || cols == 0
        || rows > 1048576
        || cols > 1048576
        || page_rows == 0
        || bytes > MAX_WEIGHT_BYTES
        || page_bytes > MAX_PAGE_BYTES
        || rows.div_ceil(page_rows) > MAX_PAGES
    {
        return Err("paged matrix exceeds bounded shape or page policy".into());
    }
    Ok((bytes, rows.div_ceil(page_rows)))
}
fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

pub fn export(
    path: &Path,
    data: &[u8],
    rows: usize,
    cols: usize,
    page_rows: usize,
    compress: bool,
) -> Result<[u8; 32], String> {
    let (size, count) = shape(rows, cols, page_rows)?;
    if data.len() != size {
        return Err("paged weight byte count differs from shape".into());
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(error)?;
    let weight_digest: [u8; 32] = Sha256::digest(data).into();
    let mut metadata = Vec::with_capacity(HEADER + count * ENTRY);
    metadata.extend_from_slice(MAGIC);
    for value in [rows, cols, page_rows, count] {
        metadata.extend_from_slice(&(value as u32).to_le_bytes());
    }
    metadata.extend_from_slice(&weight_digest);
    metadata.extend_from_slice(&[0; 8]);
    let mut offset = (HEADER + count * ENTRY) as u64;
    file.seek(SeekFrom::Start(offset)).map_err(error)?;
    for page in data.chunks(page_rows * cols) {
        let packed = if compress {
            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(page).map_err(error)?;
            encoder.finish().map_err(error)?
        } else {
            Vec::new()
        };
        let compressed = compress && packed.len() < page.len();
        let payload = if compressed { packed.as_slice() } else { page };
        metadata.extend_from_slice(&offset.to_le_bytes());
        metadata.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        metadata.extend_from_slice(&(page.len() as u32).to_le_bytes());
        metadata.push(u8::from(compressed));
        metadata.extend_from_slice(&[0; 7]);
        file.write_all(payload).map_err(error)?;
        offset += payload.len() as u64;
    }
    file.seek(SeekFrom::Start(0)).map_err(error)?;
    file.write_all(&metadata).map_err(error)?;
    file.seek(SeekFrom::Start(0)).map_err(error)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(error)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    file.as_file().sync_all().map_err(error)?;
    file.persist_noclobber(path).map_err(error)?;
    Ok(hash.finalize().into())
}

struct PageIndex {
    head: Vec<u8>,
    rows: usize,
    cols: usize,
    page_rows: usize,
    pages: Vec<Page>,
}
fn metadata(file: &mut File, length: u64) -> Result<PageIndex, String> {
    if length < HEADER as u64 || length > (MAX_WEIGHT_BYTES + HEADER + MAX_PAGES * ENTRY) as u64 {
        return Err("paged artifact byte size exceeds policy".into());
    }
    let mut head = vec![0; HEADER];
    file.read_exact(&mut head).map_err(error)?;
    if &head[..8] != MAGIC || head[56..64] != [0; 8] {
        return Err("invalid paged artifact header".into());
    }
    let get = |i: usize| u32::from_le_bytes(head[i..i + 4].try_into().unwrap()) as usize;
    let (rows, cols, page_rows, declared) = (get(8), get(12), get(16), get(20));
    let (_, count) = shape(rows, cols, page_rows)?;
    if declared != count || (HEADER + count * ENTRY) as u64 > length {
        return Err("paged index length differs".into());
    }
    head.resize(HEADER + count * ENTRY, 0);
    file.read_exact(&mut head[HEADER..]).map_err(error)?;
    let mut offset = head.len() as u64;
    let mut pages = Vec::with_capacity(count);
    for (index, entry) in head[HEADER..].chunks_exact(ENTRY).enumerate() {
        let stored = u32::from_le_bytes(entry[8..12].try_into().unwrap()) as usize;
        let decoded = u32::from_le_bytes(entry[12..16].try_into().unwrap()) as usize;
        let expected = (rows - index * page_rows).min(page_rows) * cols;
        if u64::from_le_bytes(entry[..8].try_into().unwrap()) != offset
            || decoded != expected
            || stored == 0
            || stored > decoded
            || entry[16] > 1
            || entry[17..] != [0; 7]
            || (entry[16] == 0 && stored != decoded)
            || (entry[16] == 1 && stored >= decoded)
        {
            return Err("invalid, overlapping or noncanonical paged index".into());
        }
        pages.push(Page {
            offset,
            stored,
            decoded,
            compressed: entry[16] != 0,
        });
        offset += stored as u64;
        if offset > length {
            return Err("paged artifact payload is truncated".into());
        }
    }
    if offset != length {
        return Err("paged artifact has trailing bytes".into());
    }
    Ok(PageIndex {
        head,
        rows,
        cols,
        page_rows,
        pages,
    })
}

fn read_at(file: &File, data: &mut [u8], offset: u64) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(data, offset).map_err(error)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < data.len() {
            let n = file
                .seek_read(&mut data[done..], offset + done as u64)
                .map_err(error)?;
            if n == 0 {
                return Err("private paged snapshot is truncated".into());
            }
            done += n;
        }
        Ok(())
    }
}
fn decode(file: &File, page: Page) -> Result<Vec<u8>, String> {
    let mut payload = vec![0; page.stored];
    read_at(file, &mut payload, page.offset)?;
    if !page.compressed {
        return Ok(payload);
    }
    let mut raw = vec![0; page.decoded + 1];
    let mut decoder = Decompress::new(true);
    let status = decoder
        .decompress(&payload, &mut raw, FlushDecompress::Finish)
        .map_err(error)?;
    if status != Status::StreamEnd
        || decoder.total_in() != page.stored as u64
        || decoder.total_out() != page.decoded as u64
    {
        return Err("paged zlib stream size, completion or trailing data differs".into());
    }
    raw.truncate(page.decoded);
    Ok(raw)
}

pub struct PagedMatrix {
    file: File,
    pages: Vec<Page>,
    executor: Executor,
    rows: usize,
    cols: usize,
    page_rows: usize,
    artifact_bytes: u64,
    weight_digest: [u8; 32],
    max_weight: u32,
}
impl PagedMatrix {
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub fn artifact_bytes(&self) -> u64 {
        self.artifact_bytes
    }
    pub fn weight_digest(&self) -> [u8; 32] {
        self.weight_digest
    }
    pub fn open(
        path: &Path,
        expected: [u8; 32],
        threads: usize,
        simd: bool,
    ) -> Result<Self, String> {
        if !std::fs::metadata(path).map_err(error)?.is_file() {
            return Err("paged source must be a regular file".into());
        }
        let mut source = File::open(path).map_err(error)?;
        if !source.metadata().map_err(error)?.is_file() {
            return Err("paged source must be a regular file".into());
        }
        let length = source.metadata().map_err(error)?.len();
        let PageIndex {
            head,
            rows,
            cols,
            page_rows,
            pages,
        } = metadata(&mut source, length)?;
        let executor = Executor::new(threads, simd)?;
        source.seek(SeekFrom::Start(0)).map_err(error)?;
        let mut file = tempfile::tempfile().map_err(error)?;
        let mut hash = Sha256::new();
        let mut copied = 0_u64;
        let mut buffer = [0; 65536];
        let mut limited = source.take(length + 1);
        loop {
            let n = limited.read(&mut buffer).map_err(error)?;
            if n == 0 {
                break;
            }
            file.write_all(&buffer[..n]).map_err(error)?;
            hash.update(&buffer[..n]);
            copied += n as u64;
        }
        let actual: [u8; 32] = hash.finalize().into();
        if copied != length || actual != expected {
            return Err("paged artifact digest or length differs".into());
        }
        let mut snapshot_head = vec![0; head.len()];
        read_at(&file, &mut snapshot_head, 0)?;
        if snapshot_head != head {
            return Err("paged source metadata changed during snapshot".into());
        }
        let weight_digest: [u8; 32] = head[24..56].try_into().unwrap();
        let mut raw_hash = Sha256::new();
        let mut max_weight = 0;
        for page in &pages {
            let decoded = decode(&file, *page)?;
            raw_hash.update(&decoded);
            max_weight = max_weight.max(
                decoded
                    .iter()
                    .map(|&x| (x as i8 as i32).unsigned_abs())
                    .max()
                    .unwrap_or(0),
            );
        }
        if <[u8; 32]>::from(raw_hash.finalize()) != weight_digest {
            return Err("paged decoded weight digest differs".into());
        }
        Ok(Self {
            file,
            pages,
            executor,
            rows,
            cols,
            page_rows,
            artifact_bytes: length,
            weight_digest,
            max_weight,
        })
    }
    pub fn metadata_bytes(&self) -> usize {
        self.pages.len() * std::mem::size_of::<Page>()
    }
    pub fn transient_weight_bytes(&self) -> usize {
        self.pages
            .iter()
            .map(|p| {
                if p.compressed {
                    p.stored + 2 * p.decoded + 1
                } else {
                    2 * p.decoded
                }
            })
            .max()
            .unwrap()
    }
    fn check(&self, elements: usize, batch: usize) -> Result<(), String> {
        if batch == 0
            || self.cols.checked_mul(batch) != Some(elements)
            || elements > MAX_ELEMENTS
            || self
                .rows
                .checked_mul(batch)
                .is_none_or(|n| n > MAX_ELEMENTS)
        {
            return Err("paged input/output workspace exceeds bounded shape policy".into());
        }
        Ok(())
    }
    fn execute<T: Copy + Default>(
        &self,
        batch: usize,
        mut operation: impl FnMut(&Matrix) -> Result<Vec<T>, String>,
    ) -> Result<Vec<T>, String> {
        let mut result = vec![T::default(); batch * self.rows];
        for (index, page) in self.pages.iter().enumerate() {
            let values = decode(&self.file, *page)?;
            let rows = page.decoded / self.cols;
            let matrix = if self.cols as u64 * 128 * 128 <= i32::MAX as u64 {
                Matrix::from_owned_i8_page(values, rows, self.cols)?
            } else {
                // Preserve the resident kernel's tighter admission for unusually
                // wide matrices whose actual weights have a smaller magnitude.
                Matrix::new(&values, rows, self.cols)?
            };
            let partial = operation(&matrix)?;
            for row in 0..batch {
                let first = row * self.rows + index * self.page_rows;
                result[first..first + rows].copy_from_slice(&partial[row * rows..(row + 1) * rows]);
            }
        }
        Ok(result)
    }
    pub fn clear(&self, x: &[i8], batch: usize) -> Result<Vec<i32>, String> {
        self.check(x.len(), batch)?;
        if self.cols as u64 * self.max_weight as u64 * 128 > i32::MAX as u64 {
            return Err("paged clear result exceeds int32 bound".into());
        }
        self.execute(batch, |m| m.clear(&self.executor, x, batch))
    }
    pub fn wrap32(&self, x: &[u32], batch: usize) -> Result<Vec<u32>, String> {
        self.check(x.len(), batch)?;
        self.execute(batch, |m| m.wrap32(&self.executor, x, batch))
    }
    pub fn modular(&self, x: &[u32], batch: usize, modulus: u32) -> Result<Vec<u32>, String> {
        self.check(x.len(), batch)?;
        self.execute(batch, |m| m.modular(&self.executor, x, batch, modulus))
    }
    pub fn gather(&self, ids: &[u64]) -> Result<Vec<u8>, String> {
        if ids.is_empty()
            || ids.len() > 4096
            || ids
                .len()
                .checked_mul(self.cols)
                .is_none_or(|n| n > MAX_ELEMENTS)
            || ids.iter().any(|&i| i >= self.rows as u64)
        {
            return Err("paged gather indices exceed bounds".into());
        }
        let mut grouped: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for (position, &id) in ids.iter().enumerate() {
            grouped
                .entry(id as usize / self.page_rows)
                .or_default()
                .push((position, id as usize % self.page_rows));
        }
        let mut out = vec![0; ids.len() * self.cols];
        for (page, entries) in grouped {
            let data = decode(&self.file, self.pages[page])?;
            for (position, row) in entries {
                out[position * self.cols..(position + 1) * self.cols]
                    .copy_from_slice(&data[row * self.cols..(row + 1) * self.cols]);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_pages_and_immutable_source() {
        let dir = tempfile::tempdir().unwrap();
        let w: Vec<u8> = (0..33 * 17).map(|i| (i * 37) as u8).collect();
        let reference = Matrix::new(&w, 33, 17).unwrap();
        let executor = Executor::new(1, true).unwrap();
        let x: Vec<i8> = (0..3 * 17).map(|i| (i * 7) as i8).collect();
        let u: Vec<u32> = (0..3 * 17).map(|i| u32::MAX - i as u32).collect();
        for compressed in [false, true] {
            let path = dir.path().join(format!("{compressed}.weights"));
            let digest = export(&path, &w, 33, 17, 8, compressed).unwrap();
            let matrix = PagedMatrix::open(&path, digest, 1, true).unwrap();
            std::fs::write(&path, b"changed after admission").unwrap();
            assert_eq!(
                matrix.clear(&x, 3).unwrap(),
                reference.clear(&executor, &x, 3).unwrap()
            );
            assert_eq!(
                matrix.wrap32(&u, 3).unwrap(),
                reference.wrap32(&executor, &u, 3).unwrap()
            );
            assert_eq!(
                matrix.gather(&[32, 0, 32]).unwrap(),
                [&w[32 * 17..], &w[..17], &w[32 * 17..]].concat()
            );
            assert!(matrix.gather(&[33]).is_err());
            assert!(matrix.clear(&x, 2).is_err());
        }
    }
    #[test]
    fn compressed_completion_and_resource_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights");
        let data = vec![7; 8192];
        let digest = export(&path, &data, 128, 64, 16, true).unwrap();
        let matrix = PagedMatrix::open(&path, digest, 1, false).unwrap();
        assert!(matrix.pages.iter().all(|p| p.compressed));
        assert_eq!(matrix.gather(&[0, 127]).unwrap(), vec![7; 128]);
        let p = matrix.pages[0];
        assert!(decode(
            &matrix.file,
            Page {
                stored: p.stored - 1,
                ..p
            }
        )
        .is_err());
        assert!(decode(
            &matrix.file,
            Page {
                stored: p.stored + 1,
                ..p
            }
        )
        .is_err());
        assert!(decode(
            &matrix.file,
            Page {
                decoded: p.decoded - 1,
                ..p
            }
        )
        .is_err());
        assert!(PagedMatrix::open(&path, [0; 32], 1, true).is_err());
        assert!(export(&path, &data, 128, 64, 16, false).is_err());
        assert!(shape(1048576, 1048576, 1).is_err());
        assert!(shape(16, 1048576, 8).is_err());
    }
}
