//! Test-only fused reference. No public mask API or selectable SDK path.
use super::*;
use pllm_core::kernels::{Executor, Matrix};
use serde_json::json;
use std::time::Instant;
use zeroize::Zeroizing;

#[derive(Default, Debug)]
struct Work {
    rows: usize,
    weight_values: usize,
    scale_bytes: usize,
}

struct Fused {
    server: Server,
    heads: BTreeSet<Block>,
    last_work: Work,
}

impl Fused {
    fn new(server: Server) -> Self {
        Self {
            server,
            heads: BTreeSet::new(),
            last_work: Work::default(),
        }
    }

    // All shape/layout/source/role/ticket checks precede output allocation and
    // joint burn. Invalid shape leaves original valid transaction available.
    fn admit_pair(&mut self, query: &[u8], r: &[u32], ticket: Block) -> Result<Key, String> {
        let key = Key::decode(query)?;
        let Table::Vocabulary { matrix, .. } = self.server.data.as_ref() else {
            return Err("fusion requires original tied matrix".into());
        };
        let (rows, columns) = matrix.shape();
        if key.layout != Layout::Packed128
            || key.descriptor != self.server.descriptor
            || key.party != self.server.party
            || r.len() != columns
            || rows != key.descriptor.records
            || columns + 4 != key.descriptor.width
        {
            return Err("fusion shape, packed layout, source or role differs".into());
        }
        if self.heads.len() >= MAX_QUERIES
            || self.heads.contains(&ticket)
            || self.server.burned.len() >= MAX_QUERIES
            || self.server.burned.contains(&key.id)
        {
            return Err("fusion query/head replayed or ledger exhausted".into());
        }
        // Single synchronous owner: no interleaving between these reservations.
        let admitted = self.server.admit(query)?;
        assert!(self.heads.insert(ticket));
        Ok(admitted)
    }

    fn cancel(&mut self, query: &[u8], r: &[u32], ticket: Block) -> Result<(), String> {
        self.admit_pair(query, r, ticket).map(|_| ())
    }

    fn evaluate(
        &mut self,
        query: &[u8],
        r: &[u32],
        ticket: Block,
        fail_after_rows: Option<usize>,
    ) -> Result<(Vec<u8>, Vec<u32>), String> {
        self.last_work = Work::default();
        let key = self.admit_pair(query, r, ticket)?;
        // Everything below is after joint consumption, including injected failure.
        let mut lookup = Zeroizing::new(vec![0; key.descriptor.width]);
        let mut head = Zeroizing::new(vec![0; key.descriptor.records]);
        let Table::Vocabulary { matrix, scales } = self.server.data.as_ref() else {
            unreachable!()
        };
        let mut scan = Scan {
            matrix,
            scales,
            r,
            lookup: &mut lookup,
            head: &mut head,
            work: &mut self.last_work,
            fail_after_rows,
        };
        scan.visit(
            &key,
            domain(key.descriptor, key.id, key.layout),
            key.root,
            key.party,
            0,
            0,
        )?;
        if fail_after_rows == Some(self.last_work.rows) {
            return Err("injected post-scan response failure".into());
        }
        let mut reply = header(key.layout.reply_magic(), key.descriptor, key.id, key.party);
        reply.extend_from_slice(&lookup);
        // Transfer the head allocation without making a second V-u32 copy.
        Ok((reply, std::mem::take(&mut *head)))
    }
}

struct Scan<'a> {
    matrix: &'a Matrix,
    scales: &'a [u8],
    r: &'a [u32],
    lookup: &'a mut [u8],
    head: &'a mut [u32],
    work: &'a mut Work,
    fail_after_rows: Option<usize>,
}

impl Scan<'_> {
    // Original PRG/domain/corrections; small DFS visitor, not copied DPF runtime.
    // Terminal mask stays local (16 bytes), tree stack logarithmic. Each source
    // weight value feeds XOR and ring32 arithmetic in the same inner iteration.
    #[allow(clippy::too_many_arguments)]
    fn visit(
        &mut self,
        key: &Key,
        d: Block,
        seed: Block,
        control: u8,
        level: usize,
        first: usize,
    ) -> Result<(), String> {
        let seed = Zeroizing::new(seed);
        let row_start = first * 128;
        if row_start >= key.descriptor.records {
            return Ok(());
        }
        if level == key.words.len() {
            let mut word = Zeroizing::new(leaf(*seed, d, key.layout));
            for (a, b) in word.iter_mut().zip(key.final_word) {
                *a ^= b & 0_u8.wrapping_sub(control);
            }
            let columns = self.r.len();
            for bit in 0..128.min(key.descriptor.records - row_start) {
                if self.fail_after_rows == Some(self.work.rows) {
                    return Err("injected post-admission scan failure".into());
                }
                let row = row_start + bit;
                let mask = 0_u8.wrapping_sub((word[bit / 8] >> (bit % 8)) & 1);
                let weights = &self.matrix.as_signed_slice()[row * columns..(row + 1) * columns];
                let mut dot = 0_u32;
                for (column, &weight) in weights.iter().enumerate() {
                    self.lookup[column] ^= weight as u8 & mask;
                    dot = dot.wrapping_add((weight as i32 as u32).wrapping_mul(self.r[column]));
                }
                for (out, &scale) in self.lookup[columns..]
                    .iter_mut()
                    .zip(&self.scales[4 * row..4 * row + 4])
                {
                    *out ^= scale & mask;
                }
                self.head[row] = dot;
                self.work.rows += 1;
                self.work.weight_values += columns;
                self.work.scale_bytes += 4;
            }
        } else {
            let mut children = Zeroizing::new(stretch(*seed, d, level));
            for (direction, child) in children.iter_mut().enumerate() {
                if control != 0 {
                    child.0 = xor(child.0, key.words[level].seed);
                    child.1 ^= (key.words[level].controls >> direction) & 1;
                }
                self.visit(
                    key,
                    d,
                    child.0,
                    child.1,
                    level + 1,
                    first + (direction << (key.words.len() - level - 1)),
                )?;
            }
        }
        Ok(())
    }
}

fn entropy_ring(columns: usize) -> Zeroizing<Vec<u32>> {
    let mut bytes = Zeroizing::new(vec![0; 4 * columns]);
    getrandom::fill(&mut bytes).unwrap();
    Zeroizing::new(
        bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    )
}

fn ticket() -> Block {
    let mut id = [0; 16];
    getrandom::fill(&mut id).unwrap();
    id
}

fn scalar(matrix: &Matrix, x: &[u32]) -> Vec<u32> {
    matrix
        .as_signed_slice()
        .chunks_exact(x.len())
        .map(|row| {
            row.iter().zip(x).fold(0_u32, |s, (&w, &v)| {
                s.wrapping_add((w as i32 as u32).wrapping_mul(v))
            })
        })
        .collect()
}

// A separate ledger with exactly the same immutable owner/context, exclusively
// for independent original-evaluator oracle. Never an API available to clients.
fn oracle(server: &Server) -> Server {
    Server {
        descriptor: server.descriptor,
        data: Arc::clone(&server.data),
        party: server.party,
        burned: BTreeSet::new(),
        contexts: BTreeSet::new(),
    }
}

fn geometry(rows: usize, columns: usize, index: usize, repetitions: usize) -> serde_json::Value {
    assert!((1..=4096).contains(&rows) && (1..=4096).contains(&columns));
    assert!(rows * columns <= 4_194_304 && index < rows && (1..=128).contains(&repetitions));
    let setup = Instant::now();
    let weights: Vec<u8> = (0..rows * columns)
        .map(|i| (i * 131 + i / columns) as u8)
        .collect();
    let scales: Vec<u8> = (0..rows)
        .flat_map(|i| (0.001_f32 + i as f32 / 1024.0).to_le_bytes())
        .collect();
    let matrix = Arc::new(Matrix::new(&weights, rows, columns).unwrap());
    let mut roots: Vec<_> = (0..2)
        .map(|p| Server::from_matrix(Arc::clone(&matrix), &scales, p).unwrap())
        .collect();
    let mut fused: Vec<_> = roots
        .iter_mut()
        .map(|s| Fused::new(s.bind_context([93; 32]).unwrap()))
        .collect();
    let mut separate: Vec<_> = fused.iter().map(|s| oracle(&s.server)).collect();
    assert_eq!(Arc::strong_count(&matrix), 3); // only the two Table owners
    for (a, b) in fused.iter().zip(&separate) {
        assert!(Arc::ptr_eq(&a.server.data, &b.data));
    }
    let executor = Executor::new(1, true).unwrap();
    let setup_ns = setup.elapsed().as_nanos();
    let (mut fused_ns, mut separate_ns) = (0_u128, 0_u128);
    let (mut count_weights, mut count_scales) = (0, 0);
    for iteration in 0..repetitions {
        let (qa, qb, decoder) = issue_packed(fused[0].server.descriptor(), index).unwrap();
        let queries = [&qa, &qb];
        let mut replies = Vec::new();
        for party in 0..2 {
            let r = entropy_ring(columns); // fresh role-local entropy, unrelated to DPF
            let id = ticket();
            // Balanced ordering prevents always granting one path warmer caches.
            let (result, baseline, wr);
            if (iteration + party) % 2 == 0 {
                let now = Instant::now();
                result = fused[party].evaluate(queries[party], &r, id, None).unwrap();
                fused_ns += now.elapsed().as_nanos();
                let now = Instant::now();
                baseline = separate[party]
                    .evaluate_batch(&[queries[party]])
                    .unwrap()
                    .remove(0);
                wr = matrix.wrap32(&executor, &r, 1).unwrap();
                separate_ns += now.elapsed().as_nanos();
            } else {
                let now = Instant::now();
                baseline = separate[party]
                    .evaluate_batch(&[queries[party]])
                    .unwrap()
                    .remove(0);
                wr = matrix.wrap32(&executor, &r, 1).unwrap();
                separate_ns += now.elapsed().as_nanos();
                let now = Instant::now();
                result = fused[party].evaluate(queries[party], &r, id, None).unwrap();
                fused_ns += now.elapsed().as_nanos();
            }
            assert_eq!(result.0, baseline);
            assert_eq!(
                result.0,
                oracle(&fused[party].server)
                    .evaluate(queries[party])
                    .unwrap()
            );
            assert_eq!(result.1, wr);
            assert_eq!(result.1, scalar(&matrix, &r));
            let q: Vec<_> = (0..columns)
                .map(|c| (c as i32 % 255 - 127) as u32)
                .collect();
            let complement = Zeroizing::new(
                q.iter()
                    .zip(r.iter())
                    .map(|(&q, &r)| q.wrapping_sub(r))
                    .collect::<Vec<_>>(),
            );
            let other = matrix.wrap32(&executor, &complement, 1).unwrap();
            assert_eq!(
                result
                    .1
                    .iter()
                    .zip(other)
                    .map(|(&a, b)| a.wrapping_add(b))
                    .collect::<Vec<_>>(),
                scalar(&matrix, &q)
            );
            assert_eq!(fused[party].last_work.rows, rows);
            assert_eq!(fused[party].last_work.weight_values, rows * columns);
            count_weights += fused[party].last_work.weight_values;
            count_scales += fused[party].last_work.scale_bytes;
            assert!(fused[party]
                .evaluate(queries[party], &r, ticket(), None)
                .is_err());
            replies.push(result.0);
        }
        let mut expected = weights[index * columns..(index + 1) * columns].to_vec();
        expected.extend_from_slice(&scales[4 * index..4 * index + 4]);
        assert_eq!(decoder.decode(&replies[0], &replies[1]).unwrap(), expected);
    }
    json!({"schema": "boundary-scan-fusion-probe-v2", "status": "test-only-fused-reference",
        "fused_scan_executed": true, "sdk_selectable": false, "correct": true,
        "rows": rows, "columns": columns, "index": index, "repetitions": repetitions,
        "parties_checked": 2, "setup_ns": setup_ns, "native_simd": executor.simd(),
        "fused_ns_both_parties": fused_ns, "separate_ns_both_parties": separate_ns,
        "fused_over_separate_elapsed_ratio": fused_ns as f64 / separate_ns as f64,
        "source_weight_values_visited": count_weights, "source_scale_bytes_visited": count_scales,
        "terminal_mask_bytes": 16, "full_vocabulary_mask_allocated": false,
        "head_ticket_ledger_id_bytes_per_entry": 16,
        "physical_read_instrumentation": false, "assembly_single_load_verified": false,
        "scan_claim": "one source-level traversal, not measured DRAM savings",
        "timing_scope": "test-only scalar fusion vs evaluate_batch+optimized wrap32; balanced order; no SDK or production claim"})
}

#[test]
fn fused_small_correctness() {
    for (rows, columns, index) in [
        (1, 1, 0),
        (127, 17, 126),
        (128, 16, 127),
        (129, 19, 127),
        (129, 19, 128),
        (257, 33, 0),
        (257, 33, 256),
    ] {
        let result = geometry(rows, columns, index, 1);
        assert_eq!(result["fused_scan_executed"], true);
    }
}

#[test]
fn joint_admission_failure_and_cancel() {
    let matrix = Arc::new(Matrix::new(&[255; 129 * 19], 129, 19).unwrap());
    let scales: Vec<_> = (0..129).flat_map(|_| 0.1_f32.to_le_bytes()).collect();
    for party in 0..2 {
        let mut root = Server::from_matrix(Arc::clone(&matrix), &scales, party).unwrap();
        let child = root.bind_context([4; 32]).unwrap();
        let mut f = Fused::new(child);
        let (a, b, _) = issue_packed(f.server.descriptor(), 128).unwrap();
        let query = if party == 0 { a } else { b };
        let r = entropy_ring(19);
        let id = ticket();
        assert!(f.evaluate(&query, &r[..18], id, None).is_err());
        assert!(f.server.burned.is_empty() && f.heads.is_empty());
        assert_eq!(f.last_work.rows, 0);
        let mut wrong = query.clone();
        wrong[8] = 1 - party;
        assert!(f.evaluate(&wrong, &r, id, None).is_err());
        let mut wrong = query.clone();
        wrong[17] ^= 1;
        assert!(f.evaluate(&wrong, &r, id, None).is_err());
        assert!(f.evaluate(&query[..10], &r, id, None).is_err());
        assert!(f.server.burned.is_empty() && f.heads.is_empty());
        // Partial public-prefix failure: outputs zeroize/drop, both IDs retired.
        assert!(f.evaluate(&query, &r, id, Some(7)).is_err());
        assert_eq!(f.last_work.weight_values, 7 * 19);
        assert_eq!(f.last_work.scale_bytes, 7 * 4);
        assert_eq!(f.server.burned.len(), 1);
        assert!(f.heads.contains(&id));
        assert!(f.evaluate(&query, &r, ticket(), None).is_err());
        let (a, b, _) = issue_packed(f.server.descriptor(), 128).unwrap();
        let next = if party == 0 { a } else { b };
        assert!(f.evaluate(&next, &r, id, None).is_err());
        assert_eq!(f.server.burned.len(), 1); // reused head doesn't burn fresh query
        let cancel_id = ticket();
        f.cancel(&next, &r, cancel_id).unwrap();
        assert!(f.evaluate(&next, &r, ticket(), None).is_err());
        let (a, b, _) = issue_packed(f.server.descriptor(), 128).unwrap();
        let last = if party == 0 { a } else { b };
        assert!(f.evaluate(&last, &r, cancel_id, None).is_err());
        let final_id = ticket();
        assert!(f.evaluate(&last, &r, final_id, Some(129)).is_err());
        assert_eq!(f.last_work.rows, 129);
        assert_eq!(f.last_work.weight_values, 129 * 19);
        assert!(f.heads.contains(&final_id));
        assert!(f.evaluate(&last, &r, ticket(), None).is_err());
        // Scalar namespace forbidden even though production also supports it.
        let (a, b, _) = issue(f.server.descriptor(), 128).unwrap();
        assert!(f
            .evaluate(if party == 0 { &a } else { &b }, &r, ticket(), None)
            .is_err());
    }
}

#[test]
#[ignore = "explicit bounded serial fixture; compile first and run binary directly"]
fn fused_fixture() {
    fn env(name: &str, default: usize) -> usize {
        std::env::var(name)
            .map(|s| s.parse().expect("unsigned fixture geometry"))
            .unwrap_or(default)
    }
    let rows = env("PLLM_FUSION_ROWS", 129);
    let columns = env("PLLM_FUSION_COLUMNS", 19);
    let index = env("PLLM_FUSION_INDEX", rows.saturating_sub(1));
    let reps = env("PLLM_FUSION_REPETITIONS", 1);
    println!("{}", geometry(rows, columns, index, reps));
}

#[test]
fn bounded_ledgers_and_table_preflight() {
    let matrix = Arc::new(Matrix::new(&[1, 255, 128], 1, 3).unwrap());
    let mut f = Fused::new(Server::from_matrix(matrix, &1.0_f32.to_le_bytes(), 0).unwrap());
    let (q, _, _) = issue_packed(f.server.descriptor(), 0).unwrap();
    let r = entropy_ring(3);
    for i in 0..MAX_QUERIES {
        let mut id = [0; 16];
        id[..8].copy_from_slice(&(i as u64).to_le_bytes());
        f.heads.insert(id);
    }
    assert!(f.evaluate(&q, &r, ticket(), None).is_err());
    assert!(f.server.burned.is_empty());
    assert_eq!(f.last_work.weight_values, 0);
    f.server.burned = std::mem::take(&mut f.heads);
    assert!(f.evaluate(&q, &r, ticket(), None).is_err());
    assert!(f.heads.is_empty());
    assert_eq!(f.last_work.rows, 0);
    let mut raw = Fused::new(Server::new(&[1, 255, 128, 0, 0, 128, 63], 7, 0).unwrap());
    let (q, _, _) = issue_packed(raw.server.descriptor(), 0).unwrap();
    assert!(raw.evaluate(&q, &r, ticket(), None).is_err());
    assert!(raw.server.burned.is_empty() && raw.heads.is_empty());
}
