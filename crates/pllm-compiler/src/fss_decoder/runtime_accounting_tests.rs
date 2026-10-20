//! Small operator schedules isolate read accounting from compiler construction.
//! The zero-output Driver below tests schedule/resource parity, not cryptography.
use super::*;
use crate::fss_decoder::source::TensorImport;

fn tensor(rows: usize, width: usize, file: bool) -> Tensor {
    let words: Vec<_> = (0..rows * width)
        .map(|i| ((i as i32 % 7 - 3) as f32 / 8.0).to_bits())
        .collect();
    if file {
        let mut import = TensorImport::new("weight", &[rows, width], Capacity::Decoder).unwrap();
        let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        for chunk in bytes.chunks(super::super::capacity::SOURCE_READ_BYTES) {
            import.append(WeightDtype::F32, chunk).unwrap();
        }
        import.finish().unwrap()
    } else {
        Tensor::from_ieee("weight", &[rows, width], WeightDtype::F32, &words).unwrap()
    }
}

fn embedding(
    capacity: Capacity,
    file: bool,
    positions: bool,
    rows: usize,
    width: usize,
    initial: usize,
    selections: usize,
) -> Program {
    let tensor = tensor(rows, width, file);
    let range = tensor.range;
    let source = Source::new_with_capacity(vec![tensor], capacity).unwrap();
    let sequence = initial + selections - 1;
    let resources = Resources {
        metadata_bytes: source.reservation_bytes() + 4096,
        live_array_bytes_per_party: 4096,
        ..Resources::default()
    };
    Program {
        source,
        nodes: vec![
            Node {
                code: Code::Embedding {
                    weight: 0,
                    positions,
                },
                shape: vec![1, sequence, width],
                range,
                last_use: 1,
            },
            Node {
                code: Code::Last(0),
                shape: vec![1, width],
                range,
                last_use: 2,
            },
            Node {
                code: Code::Argmax(1),
                shape: vec![1],
                range: Range::new(0, rows as i128 - 1).unwrap(),
                last_use: 2,
            },
        ],
        initial,
        selections,
        sequence,
        vocabulary: rows,
        digest: [71; 32],
        obligations: [0; 32],
        resources,
        limits: Limits::for_capacity(capacity),
        energy: None,
    }
}

fn context(p: &Program) -> Context {
    Context {
        plan: p.digest,
        session: [72; 32],
        operation: [73; 32],
        first_tensor_index: 0,
    }
}

#[derive(Default)]
struct ZeroDriver {
    calls: usize,
}
impl Driver for ZeroDriver {
    fn run(&mut self, _: Context, plan: &Plan<'_>, _: &[u64]) -> Result<NativeOutput, Error> {
        self.calls += 1;
        let count = plan.output_count();
        Ok(NativeOutput {
            values: Zeroizing::new(zeros(count)?),
        })
    }
    fn cancel(&mut self) {}
}

fn walk(p: &Program, role: u8, driven: bool) -> (Resources, [u8; 32]) {
    let budget = request_budget(p).unwrap();
    let tables = request_tables(p, &budget).unwrap();
    let mut driver = ZeroDriver::default();
    let mut state = Schedule::new(p, context(p), role, None, &tables).unwrap();
    if driven {
        state.driver = Some(&mut driver);
    }
    assert_eq!(
        state.execute(&vec![0; p.initial]).unwrap().len(),
        p.selections
    );
    (state.resources, state.hash.clone().finalize().into())
}

#[test]
fn embedding_batches_complete_padded_scans_in_preflight_and_each_holder_walk() {
    // 3 selections each evaluate 4 rows, NOT growing prefixes 2+3+4.
    // Each bounded batch scans all 7*3 i64 coefficients once at both holders:
    // three batches cost 1008 bytes, rather than twelve separate scans (4032).
    let resident = embedding(Capacity::Decoder, false, false, 7, 3, 2, 3);
    let file = embedding(Capacity::Decoder, true, false, 7, 3, 2, 3);
    assert_eq!(resident.source.digest(), file.source.digest());
    for role in [0, 1] {
        for driven in [false, true] {
            let (a, ah) = walk(&resident, role, driven);
            let (b, bh) = walk(&file, role, driven);
            let a_copies = a.public_copy_bytes - 3 * resident.source.reservation_bytes();
            let b_copies = b.public_copy_bytes - 3 * file.source.reservation_bytes();
            assert_eq!(b_copies - a_copies, 1008);
            assert_ne!(b_copies - a_copies, 4032);
            assert_eq!(ah, bh); // Accounting cannot change kernel obligations.
            assert_eq!(a.kernels, b.kernels);
            assert_eq!(a.material_bytes, b.material_bytes);
            assert_eq!(a.frame_bytes, b.frame_bytes);
        }
    }
    let mut file = file;
    let (expected, hash) = walk(&file, 0, false);
    preflight(&mut file).unwrap();
    assert_eq!(file.resources.public_copy_bytes, expected.public_copy_bytes);
    assert_eq!(file.obligations, hash);
}

#[test]
fn embedding_limit_rejects_before_even_the_first_kernel_or_file_scan() {
    for driven in [false, true] {
        let mut p = embedding(Capacity::Decoder, true, false, 17, 9, 2, 1);
        let base = 3 * p.source.reservation_bytes();
        // Two padded rows share one 17*9 coefficient scan on each holder.
        p.limits.max_public_copy_bytes = base + 2448 - 1;
        let budget = request_budget(&p).unwrap();
        let tables = request_tables(&p, &budget).unwrap();
        let mut driver = ZeroDriver::default();
        {
            let mut state = Schedule::new(&p, context(&p), 0, None, &tables).unwrap();
            if driven {
                state.driver = Some(&mut driver);
            }
            assert!(matches!(
                state.node(&p.nodes[0], &[], &[0, 0], 2),
                Err(Error::Resource)
            ));
            assert_eq!(state.resources.kernels, 0);
        }
        assert_eq!(driver.calls, 0);
    }
    // Independent native kernel overhead makes a nonzero control allowance.
    let mut p = embedding(Capacity::Decoder, true, false, 17, 9, 2, 1);
    let copy_overhead = 2 * 3 * std::mem::size_of::<cache::Stats>();
    p.limits.max_public_copy_bytes = 3 * p.source.reservation_bytes() + 2448 + copy_overhead;
    let budget = request_budget(&p).unwrap();
    let tables = request_tables(&p, &budget).unwrap();
    let mut state = Schedule::new(&p, context(&p), 0, None, &tables).unwrap();
    state.node(&p.nodes[0], &[], &[0, 0], 2).unwrap();
    assert_eq!(
        state.resources.public_copy_bytes,
        p.limits.max_public_copy_bytes
    );
}

#[test]
fn embedding_bounded_batch_tail_preserves_every_selected_row_and_fresh_kernel() {
    // Geometry regression, not a trained-model fixture. Five rows require two
    // scans at this vocabulary; the final one-row batch must neither reuse a
    // prior selector nor overwrite any earlier output. Only this node executes.
    let vocabulary = Capacity::Decoder.max_vocabulary();
    let mut p = embedding(Capacity::Decoder, true, false, vocabulary, 1, 5, 1);
    assert_eq!(p.embedding_batch_rows().unwrap(), 4);
    p.resources.live_array_bytes_per_party =
        4 * vocabulary * 8 + 2 * super::super::capacity::SOURCE_READ_BYTES + 4096;
    struct ClearSelectors {
        invocations: Vec<Context>,
    }
    impl Driver for ClearSelectors {
        fn run(
            &mut self,
            context: Context,
            plan: &Plan<'_>,
            input: &[u64],
        ) -> Result<NativeOutput, Error> {
            assert_eq!(input.len(), 1);
            assert_eq!(plan.resources().secret_steps, 1);
            self.invocations.push(context);
            let mut values = Zeroizing::new(zeros(plan.output_count())?);
            values[input[0] as usize] = 1;
            Ok(NativeOutput { values })
        }
        fn cancel(&mut self) {}
    }
    let input = [0, 1, vocabulary as u64 - 1, 17, 12345];
    let budget = request_budget(&p).unwrap();
    let tables = request_tables(&p, &budget).unwrap();
    let mut driver = ClearSelectors {
        invocations: Vec::new(),
    };
    let mut state = Schedule::new(&p, context(&p), 0, Some(&mut driver), &tables).unwrap();
    let before = state.resources.public_copy_bytes;
    let actual = state.node(&p.nodes[0], &[], &input, 5).unwrap();
    let expected = input.map(|i| word((i as i64 % 7 - 3) * 512));
    assert_eq!(actual.as_slice(), expected);
    assert_eq!(state.resources.kernels, 5);
    assert_eq!(
        state.resources.public_copy_bytes - before,
        2 * vocabulary * 16 + 5 * 3 * std::mem::size_of::<cache::Stats>()
    );
    drop(state);
    assert_eq!(driver.invocations.len(), 5);
    for pair in driver.invocations.windows(2) {
        assert_ne!(pair[0].operation, pair[1].operation);
    }
}

#[test]
fn position_reads_price_only_selected_rows_and_role_zero() {
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        let p = embedding(capacity, capacity == Capacity::Decoder, true, 7, 3, 2, 1);
        for role in [0, 1] {
            let budget = request_budget(&p).unwrap();
            let tables = request_tables(&p, &budget).unwrap();
            let mut state = Schedule::new(&p, context(&p), role, None, &tables).unwrap();
            let before = state.resources.public_copy_bytes;
            let actual = state.node(&p.nodes[0], &[], &[0, 0], 2).unwrap();
            let expected = if role == 0 && capacity == Capacity::Decoder {
                48
            } else {
                0
            };
            assert_eq!(state.resources.public_copy_bytes - before, expected);
            if role == 0 {
                assert_eq!(*actual, [-1536, -1024, -512, 0, 512, 1024].map(word));
            } else {
                assert_eq!(*actual, [0; 6]);
            }
        }
    }
}

#[test]
fn position_limit_and_shape_control_establish_charge_before_reads() {
    let mut p = embedding(Capacity::Decoder, true, true, 1, 3, 2, 1);
    // Deliberately invalid second public position: without a prior quota check,
    // row zero is read and the second row returns Domain instead of Resource.
    let base = 3 * p.source.reservation_bytes();
    p.limits.max_public_copy_bytes = base + 47;
    let run = |p: &Program| {
        let budget = request_budget(p).unwrap();
        let tables = request_tables(p, &budget).unwrap();
        let mut state = Schedule::new(p, context(p), 0, None, &tables).unwrap();
        state.node(&p.nodes[0], &[], &[0, 0], 2).map(|_| ())
    };
    assert_eq!(run(&p), Err(Error::Resource));
    p.limits.max_public_copy_bytes = base + 48;
    assert_eq!(run(&p), Err(Error::Domain));
}

fn linear(rows: usize, inner: usize, columns: usize) -> Program {
    let mut p = embedding(Capacity::Decoder, true, false, columns, inner, rows, 1);
    p.nodes[1] = Node {
        code: Code::Linear {
            input: 0,
            weight: 0,
            bias: None,
            input_output: false,
        },
        shape: vec![1, rows, columns],
        range: Range::new(-4096, 4096).unwrap(),
        last_use: 2,
    };
    p
}

#[test]
fn linear_charges_one_pair_of_scans_and_three_statistics_walks() {
    for rows in [1, 3] {
        let p = linear(rows, 5, 2);
        for driven in [false, true] {
            let budget = request_budget(&p).unwrap();
            let tables = request_tables(&p, &budget).unwrap();
            let mut driver = ZeroDriver::default();
            let mut state = Schedule::new(&p, context(&p), 0, None, &tables).unwrap();
            if driven {
                state.driver = Some(&mut driver);
            }
            let arrays = [Some(Zeroizing::new(vec![0; rows * 5]))];
            // Direct node walks set the same consumer cursor as execute().
            state.node = 1;
            state.node(&p.nodes[1], &arrays, &[], rows).unwrap();

            // Independent control executes the same number of scalar ARS12
            // kernels (range endpoints do not affect their copy cost). Remove
            // their ordinary metadata cost without using a new read-accounting
            // helper or the production aggregate total.
            let mut control = Schedule::new(&p, context(&p), 0, None, &tables).unwrap();
            control.node = 1;
            let binding = control.binding();
            for _ in 0..rows * 2 {
                control
                    .kernel(&[0], 1, |b| {
                        Ok(kernels::floor(binding, Range::new(0, 0)?.gate(), 12, b)?.0)
                    })
                    .unwrap();
            }
            // The coefficient file is 80 bytes: two holders scan it once even
            // for three input rows. Each of two columns needs one 48-byte
            // certificate on two workers plus dealer, reused for all rows.
            assert_eq!(
                state.resources.public_copy_bytes - control.resources.public_copy_bytes,
                160 + 288,
            );
        }
    }
}

#[test]
fn linear_limit_precedes_contraction_and_certificate_reads() {
    let mut p = linear(3, 5, 2);
    let base = 3 * p.source.reservation_bytes();
    p.limits.max_public_copy_bytes = base + 448 - 1;
    let run = |p: &Program| {
        let budget = request_budget(p).unwrap();
        let tables = request_tables(p, &budget).unwrap();
        let mut driver = ZeroDriver::default();
        let mut state = Schedule::new(p, context(p), 0, Some(&mut driver), &tables).unwrap();
        state.node = 1;
        // The contraction wrapper rejects this incomplete input row before its
        // own file read. A quota failure must precede even that wrapper call.
        let arrays = [Some(Zeroizing::new(vec![0; 16]))];
        state.node(&p.nodes[1], &arrays, &[], 3).map(|_| ())
    };
    assert_eq!(run(&p), Err(Error::Resource));
    p.limits.max_public_copy_bytes = base + 448;
    assert_eq!(run(&p), Err(Error::Context));
}

#[test]
fn capsule_lookup_copy_cost_remains_its_existing_kernel_metadata_only() {
    let p = embedding(Capacity::Capsule, false, false, 7, 3, 2, 3);
    let (r, hash) = walk(&p, 0, false);
    // Four one-hot kernels and one argmax per selection, no source-table lookup.
    assert_eq!(r.kernels, 15);
    assert_eq!(
        r.public_copy_bytes,
        3 * p.source.reservation_bytes() + 15 * 3 * std::mem::size_of::<cache::Stats>(),
    );
    let (driven, driven_hash) = walk(&p, 0, true);
    assert_eq!(r.public_copy_bytes, driven.public_copy_bytes);
    assert_eq!(hash, driven_hash);
}

#[test]
fn copy_charge_checked_addition_rejects_overflow() {
    let p = embedding(Capacity::Capsule, false, false, 7, 3, 1, 1);
    let budget = request_budget(&p).unwrap();
    let tables = request_tables(&p, &budget).unwrap();
    let mut state = Schedule::new(&p, context(&p), 0, None, &tables).unwrap();
    state.resources.public_copy_bytes = usize::MAX;
    assert_eq!(state.public_copy(1), Err(Error::Resource));
}
