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
        online: OnlineSchedule::Serial,
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
fn shared_phase_geometry_preserves_active_masks_last_row_feedback_and_padded_walk() {
    // Synthetic clear Driver only: no keys, RNG or cryptographic claims. Every
    // phase still runs four padded rows. These masks/rows are enumerated without
    // calling the production geometry helper or reproducing its arithmetic.
    let mut p = embedding(Capacity::Capsule, false, false, 7, 4, 2, 3);
    p.nodes = vec![
        Node {
            code: p.nodes[0].code, shape: p.nodes[0].shape.clone(),
            range: p.nodes[0].range, last_use: 2,
        },
        Node {
            code: Code::Softmax(0), shape: vec![1, 1, 4, 4],
            range: Range::new(0, 4096).unwrap(), last_use: 2,
        },
        Node {
            code: Code::Last(0), shape: vec![1, 4],
            range: p.nodes[0].range, last_use: 3,
        },
        Node {
            code: Code::Argmax(2), shape: vec![1],
            range: Range::new(0, 6).unwrap(), last_use: 3,
        },
    ];
    let masks = [
        [true, false, false, false], [true, true, false, false],
        [true, true, false, false], [true, true, false, false],
        [true, false, false, false], [true, true, false, false],
        [true, true, true, false], [true, true, true, false],
        [true, false, false, false], [true, true, false, false],
        [true, true, true, false], [true, true, true, true],
    ];
    let reference_budget = request_budget(&p).unwrap();
    let reference_tables = request_tables(&p, &reference_budget).unwrap();
    let rs = ranges(4, p.nodes[0].range).unwrap();
    let expected_masks: Vec<[u8; 32]> = masks.iter().map(|mask| {
        gates::softmax_cached(&rs, mask, &reference_tables).unwrap().0.digest()
    }).collect();
    assert_ne!(expected_masks[1], expected_masks[6]);
    assert_ne!(expected_masks[6], expected_masks[11]);
    drop(reference_tables);
    drop(reference_budget);

    struct GeometryDriver {
        masks: Vec<[u8; 32]>,
        softmax_calls: usize,
        selections: usize,
        observations: usize,
    }
    impl Driver for GeometryDriver {
        fn run(&mut self, _: Context, plan: &Plan<'_>, input: &[u64]) -> Result<NativeOutput, Error> {
            let count = plan.output_count();
            let mut values = Zeroizing::new(zeros(count)?);
            match count {
                7 => { // Clear one-hot selector, used only in this synthetic fixture.
                    assert_eq!(input.len(), 1);
                    values[input[0] as usize] = 1;
                }
                4 => {
                    assert_eq!(plan.digest(), self.masks[self.softmax_calls]);
                    self.softmax_calls += 1;
                }
                1 => {
                    values[0] = [4, 5, 6][self.selections];
                    self.selections += 1;
                }
                _ => panic!("unexpected synthetic kernel"),
            }
            Ok(NativeOutput { values })
        }
        fn observe(&mut self, selection: usize, node: usize, shape: &[usize], values: &[u64]) {
            self.observations += 1;
            // Exact public coefficient rows from the small tensor() fixture.
            // Token feedback must replace row 2, then row 3, not the padded end
            // in every phase. Last must select rows 1, 2, 3 respectively.
            let first = [512, 1024, 1536, -1536].map(word);
            let initial_last = [-1024, -512, 0, 512].map(word);
            let unused = [-1536, -1024, -512, 0].map(word);
            let feedback_a = [-512, 0, 512, 1024].map(word);
            let feedback_b = [1536, -1536, -1024, -512].map(word);
            if node == 0 {
                assert_eq!(shape, [1, 4, 4]);
                let rows = match selection {
                    0 => [first, initial_last, unused, unused],
                    1 => [first, initial_last, feedback_a, unused],
                    2 => [first, initial_last, feedback_a, feedback_b],
                    _ => panic!("unexpected selection"),
                };
                assert_eq!(values, rows.as_flattened());
            } else if node == 2 {
                assert_eq!(values, [initial_last, feedback_a, feedback_b][selection]);
            }
        }
        fn cancel(&mut self) {}
    }
    let (expected, obligations) = walk(&p, 0, false);
    assert_eq!(expected.kernels, 27); // 3 * (4 one-hots + 4 softmaxes + 1 argmax).
    let budget = request_budget(&p).unwrap();
    let tables = request_tables(&p, &budget).unwrap();
    let mut driver = GeometryDriver {
        masks: expected_masks, softmax_calls: 0, selections: 0, observations: 0,
    };
    let mut schedule = Schedule::new(&p, context(&p), 0, Some(&mut driver), &tables).unwrap();
    assert_eq!(schedule.execute(&[1, 2]).unwrap().as_slice(), [4, 5, 6]);
    assert_eq!(resource_fields(schedule.resources), resource_fields(expected));
    assert_eq!(<[u8; 32]>::from(schedule.hash.clone().finalize()), obligations);
    drop(schedule);
    assert_eq!((driver.softmax_calls, driver.selections, driver.observations), (12, 3, 12));
    for role in [0, 1] {
        for driven in [false, true] {
            let (actual, hash) = walk(&p, role, driven);
            assert_eq!(resource_fields(actual), resource_fields(expected));
            assert_eq!(hash, obligations);
        }
    }
    preflight(&mut p).unwrap();
    assert_eq!(resource_fields(p.resources), resource_fields(expected));
    assert_eq!(p.obligations, obligations);
}

#[test]
fn schedule_rejects_inconsistent_phase_extent_before_first_kernel() {
    for (initial, selections, sequence) in [(0, 3, 4), (2, 0, 4), (2, 3, 3), (usize::MAX, 2, 4)] {
        let mut p = embedding(Capacity::Capsule, false, false, 7, 1, 2, 3);
        p.initial = initial;
        p.selections = selections;
        p.sequence = sequence;
        let budget = request_budget(&p).unwrap();
        let tables = request_tables(&p, &budget).unwrap();
        let mut driver = ZeroDriver::default();
        let mut schedule = Schedule::new(&p, context(&p), 0, Some(&mut driver), &tables).unwrap();
        assert!(matches!(schedule.execute(&[0, 0]), Err(Error::Domain)));
        assert_eq!(schedule.resources.kernels, 0);
        drop(schedule);
        assert_eq!(driver.calls, 0);
    }
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

// Supplied-record census: no Plan/cache construction or native execution here.
fn quote_seed() -> Resources {
    Resources {
        kernels: 2, primitive_requests_per_party: 3, work: 5,
        material_bytes: 7, frame_bytes: 11, public_copy_bytes: 13,
        metadata_bytes: 17, live_array_bytes_per_party: 19,
        source_table_cache_bytes_per_role: 23, peak_native_bytes: 29,
        combined_reservation_bytes: 31,
    }
}
fn quote_native() -> pllm_garble::fusefss_backend::composition::Resources {
    pllm_garble::fusefss_backend::composition::Resources {
        instructions: 2, secret_steps: 3, cumulative_work: 5,
        private_material_bytes_total: 7, frame_bytes_total: 11,
        public_table_copy_bytes_total: 13, plan_reservation_bytes: 17,
        template_construction_bytes: 19, template_copy_bytes: 23,
        public_table_reservation_bytes: 2049, combined_reservation_bytes: 29,
        ..Default::default()
    }
}
fn quote_wave() -> wavefront::Resources {
    wavefront::Resources {
        construction_work: 31, metadata_bytes: 37,
        frame_bytes_per_party: 41, additional_reservation_bytes: 43,
        ..Default::default()
    }
}
fn quote_stats() -> cache::Stats {
    cache::Stats {
        references: 3, hits: 2, misses: 1, table_payload_copy_bytes: 7,
        lowered_construction_metadata_bytes: 7,
        retained_bytes: 47, metadata_reservation_bytes: 53, retained_entries: 2,
        ..Default::default()
    }
}
fn resource_fields(r: Resources) -> [usize; 11] {
    [r.kernels, r.primitive_requests_per_party, r.work, r.material_bytes,
        r.frame_bytes, r.public_copy_bytes, r.metadata_bytes,
        r.live_array_bytes_per_party, r.source_table_cache_bytes_per_role,
        r.peak_native_bytes, r.combined_reservation_bytes]
}

fn backing_plan<'b>(p: &Program, budget: &'b Budget, tensor: bool) -> Plan<'b> {
    let plan = kernels::floor(kernels::Binding {
        source_digest: p.source.digest(), contract_digest: p.digest,
    }, Range::new(-100_000, 100_000).unwrap().gate(), 12, budget).unwrap().0;
    if tensor { plan.repeat_tensor(17).unwrap() } else { plan }
}

fn lookup_backing_plan(budget: &Budget, tensor: bool) -> Plan<'_> {
    let plan = gates::silu(gates::InputRange { min: 0, max: 4096 }, budget).unwrap().0;
    assert!(plan.stored_lookup_chunks() > 0);
    if tensor { plan.repeat_tensor(3).unwrap() } else { plan }
}

#[test]
fn wave_lookup_tariff_is_additive_and_refusal_precedes_constructor_on_same_ledger() {
    for tensor in [false, true] {
        for meter in 0..4 {
            let mut p = embedding(Capacity::Capsule, false, false, 2, 2, 1, 1);
            p.online = OnlineSchedule::WavefrontV1;
            let budget = request_budget(&p).unwrap();
            let tables = request_tables(&p, &budget).unwrap();
            let plan = lookup_backing_plan(&budget, tensor);
            let backing = wavefront::Schedule::backing_resources(&plan).unwrap();
            let delta = 6 * (128 * plan.stored_lookup_chunks());
            assert_eq!(delta, 768 * plan.stored_lookup_chunks());
            let seed = tables.bootstrap;
            let prefix = seed.work + 6 * backing.construction_work;
            p.limits.max_work = prefix + delta;
            p.limits.max_public_copy_bytes = seed.public_copy_bytes + 6 * backing.metadata_bytes;
            // Original work/copy prefix refusals precede the new debit. A
            // former exact prefix-only fit now refuses after spending delta.
            if meter == 1 { p.limits.max_work = prefix - 1; }
            if meter == 2 { p.limits.max_public_copy_bytes -= 1; }
            if meter == 3 { p.limits.max_work = prefix; }
            let mut state = Schedule::new(&p, context(&p), 0, None, &tables).unwrap();
            let hash = <[u8; 32]>::from(state.hash.clone().finalize());
            let before = budget.used_bytes();
            let input = vec![0; plan.input_bits().len()];
            let (result, entries) = wave_constructor_probe::scoped(|| {
                state.run(&plan, &input, plan.output_count(), false).map(drop)
            });
            assert_eq!(result, Err(if meter == 0 { Error::Unsupported } else { Error::Resource }));
            assert_eq!(entries, usize::from(meter == 0));
            let expected_work = prefix + if meter == 0 || meter == 3 { delta } else { 0 };
            assert_eq!(resource_fields(state.resources), resource_fields(Resources {
                work: expected_work,
                public_copy_bytes: seed.public_copy_bytes + 6 * backing.metadata_bytes,
                ..seed
            }));
            assert_eq!(<[u8; 32]>::from(state.hash.clone().finalize()), hash);
            assert_eq!(budget.used_bytes(), before);
            // Retry borrows the SAME original spent counters, not bootstrap.
            let (retry, entries) = wave_constructor_probe::scoped(|| {
                state.run(&plan, &input, plan.output_count(), false).map(drop)
            });
            assert_eq!(retry, Err(Error::Resource));
            assert_eq!(entries, 0);
            assert_eq!(state.resources.work, expected_work + 6 * backing.construction_work);
        }
    }
}

#[test]
fn wave_lookup_constructor_failure_keeps_delta_and_residual_never_subtracts_it() {
    for tensor in [false, true] {
        let mut p = embedding(Capacity::Capsule, false, false, 2, 2, 1, 1);
        p.online = OnlineSchedule::WavefrontV1;
        let budget = request_budget(&p).unwrap();
        let tables = request_tables(&p, &budget).unwrap();
        let plan = lookup_backing_plan(&budget, tensor);
        let backing = wavefront::Schedule::backing_resources(&plan).unwrap();
        let delta = 768 * plan.stored_lookup_chunks();
        let limit = resource_quote::Quote::available_per_role(
            p.resources.metadata_bytes, p.resources.live_array_bytes_per_party).unwrap();
        let occupied = budget.charge_external(limit - budget.used_bytes()
            - backing.metadata_bytes - backing.depth_bytes + 1
            - std::mem::size_of::<pllm_garble::shared_allocation::ExternalCharge<'_>>()).unwrap();
        let before = budget.used_bytes();
        let mut state = Schedule::new(&p, context(&p), 0, None, &tables).unwrap();
        let seed = state.resources;
        let input = vec![0; plan.input_bits().len()];
        assert!(matches!(state.run(&plan, &input, plan.output_count(), false), Err(Error::Resource)));
        assert_eq!(resource_fields(state.resources), resource_fields(Resources {
            work: seed.work + 6 * backing.construction_work + delta,
            public_copy_bytes: seed.public_copy_bytes + 6 * backing.metadata_bytes,
            ..seed
        }));
        assert_eq!(budget.used_bytes(), before);
        // Retry the same actual constructor with all original owners and the
        // scarcity charge still live. Both debits fit unchanged meter caps, so
        // refusal remains native reservation scarcity, not preadmission limits.
        let retried = Resources {
            work: seed.work + 2 * (6 * backing.construction_work + delta),
            public_copy_bytes: seed.public_copy_bytes + 12 * backing.metadata_bytes,
            ..seed
        };
        assert!(retried.work <= p.limits.max_work);
        assert!(retried.public_copy_bytes <= p.limits.max_public_copy_bytes);
        assert!(matches!(state.run(&plan, &input, plan.output_count(), false), Err(Error::Resource)));
        assert_eq!(resource_fields(state.resources), resource_fields(retried));
        assert_eq!(budget.used_bytes(), before);
        drop(state);
        drop(occupied);
        let mut resources = quote_seed();
        let seed = resources;
        let prefix = resource_quote::Quote::wave_backing(
            &mut resources, Limits::default(), &plan).unwrap();
        assert_eq!(prefix, backing.construction_work); // Not W0 + metadata.
        let schedule = wavefront::Schedule::new(&plan).unwrap();
        let wave = schedule.resources();
        let native = plan.resources();
        drop(resource_quote::Quote::begin(&mut resources, Limits::default(),
            Capacity::Capsule, native, Some((wave, prefix)), plan.input_bits().len(), false).unwrap());
        assert_eq!(resources.work, seed.work + 6 * wave.construction_work + delta
            + 2 * native.cumulative_work + 3 * native.template_copy_bytes);
        assert_eq!(resources.public_copy_bytes, seed.public_copy_bytes
            + 6 * wave.metadata_bytes + 3 * native.template_copy_bytes);
        drop(schedule);
        drop(plan);
        drop(tables);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn wave_backing_prefix_refuses_before_constructor_and_exact_enters() {
    for tensor in [false, true] {
        for meter in 0..3 {
            let mut p = embedding(Capacity::Capsule, false, false, 2, 2, 1, 1);
            p.online = OnlineSchedule::WavefrontV1;
            let budget = request_budget(&p).unwrap();
            let tables = request_tables(&p, &budget).unwrap();
            let plan = backing_plan(&p, &budget, tensor);
            assert_eq!(plan.stored_lookup_chunks(), 0); // Zero delta, not zero constructor work.
            let backing = wavefront::Schedule::backing_resources(&plan).unwrap();
            assert!(backing.construction_work > 0);
            let seed = tables.bootstrap;
            p.limits.max_work = seed.work + 6 * backing.construction_work;
            p.limits.max_public_copy_bytes = seed.public_copy_bytes + 6 * backing.metadata_bytes;
            if meter == 1 { p.limits.max_work -= 1; }
            if meter == 2 { p.limits.max_public_copy_bytes -= 1; }
            let mut state = Schedule::new(&p, context(&p), 0, None, &tables).unwrap();
            let hash = <[u8; 32]>::from(state.hash.clone().finalize());
            let before = budget.used_bytes();
            let peak = budget.admitted_peak_bytes();
            let input = vec![0; plan.input_bits().len()];
            let (result, entries) = wave_constructor_probe::scoped(|| {
                state.run(&plan, &input, plan.output_count(), false).map(drop)
            });
            assert_eq!(result, Err(if meter == 0 { Error::Unsupported } else { Error::Resource }));
            assert_eq!(entries, usize::from(meter == 0));
            let expected = Resources {
                work: seed.work + 6 * backing.construction_work,
                public_copy_bytes: seed.public_copy_bytes + 6 * backing.metadata_bytes,
                ..seed
            };
            assert_eq!(resource_fields(state.resources), resource_fields(expected));
            assert_eq!(<[u8; 32]>::from(state.hash.clone().finalize()), hash);
            assert_eq!(budget.used_bytes(), before);
            assert_eq!(budget.admitted_peak_bytes(), peak);
        }
    }
}

#[test]
fn wave_backing_native_construction_failure_keeps_prefix_spent() {
    let mut p = embedding(Capacity::Capsule, false, false, 2, 2, 1, 1);
    p.online = OnlineSchedule::WavefrontV1;
    let budget = request_budget(&p).unwrap();
    let tables = request_tables(&p, &budget).unwrap();
    let plan = backing_plan(&p, &budget, true);
    let backing = wavefront::Schedule::backing_resources(&plan).unwrap();
    let limit = resource_quote::Quote::available_per_role(
        p.resources.metadata_bytes, p.resources.live_array_bytes_per_party).unwrap();
    // Force scratch reservation to fail AFTER metadata reservation succeeds.
    // Occupancy belongs to the same original budget and is never reborrowed.
    let occupied = budget.charge_external(limit - budget.used_bytes()
        - backing.metadata_bytes - backing.depth_bytes + 1
        - std::mem::size_of::<pllm_garble::shared_allocation::ExternalCharge<'_>>()).unwrap();
    let before = budget.used_bytes();
    let mut state = Schedule::new(&p, context(&p), 0, None, &tables).unwrap();
    let seed = state.resources;
    let hash = <[u8; 32]>::from(state.hash.clone().finalize());
    let input = vec![0; plan.input_bits().len()];
    assert!(matches!(state.run(&plan, &input, plan.output_count(), false), Err(Error::Resource)));
    assert_eq!(resource_fields(state.resources), resource_fields(Resources {
        work: seed.work + 6 * backing.construction_work,
        public_copy_bytes: seed.public_copy_bytes + 6 * backing.metadata_bytes,
        ..seed
    }));
    assert_eq!(<[u8; 32]>::from(state.hash.clone().finalize()), hash);
    assert_eq!(budget.used_bytes(), before);
    drop(state);
    drop(occupied);
    drop(plan);
    drop(tables);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn wave_backing_overflow_precedes_constructor_and_residual_is_once_only() {
    let p = embedding(Capacity::Capsule, false, false, 2, 2, 1, 1);
    let budget = request_budget(&p).unwrap();
    for tensor in [false, true] {
        let plan = backing_plan(&p, &budget, tensor);
        let backing = wavefront::Schedule::backing_resources(&plan).unwrap();
        let before = budget.used_bytes();
        for copy in [false, true] {
            let mut resources = quote_seed();
            if copy { resources.public_copy_bytes = usize::MAX; }
            else { resources.work = usize::MAX; }
            let seed = resources;
            let mut entries = 0;
            let result = (|| {
                resource_quote::Quote::wave_backing(&mut resources, Limits::default(), &plan)?;
                entries += 1;
                wavefront::Schedule::new(&plan).map(drop).map_err(Error::from)
            })();
            assert_eq!(result, Err(Error::Resource));
            assert_eq!(entries, 0);
            assert_eq!(resources.work, if copy {
                seed.work + 6 * backing.construction_work
            } else { usize::MAX });
            assert_eq!(resources.public_copy_bytes, seed.public_copy_bytes);
            assert_eq!(budget.used_bytes(), before);
        }
        let mut resources = quote_seed();
        let seed = resources;
        let prefix = resource_quote::Quote::wave_backing(
            &mut resources, Limits::default(), &plan).unwrap();
        let schedule = wavefront::Schedule::new(&plan).unwrap();
        let wave = schedule.resources();
        let native = plan.resources();
        let q = resource_quote::Quote::begin(&mut resources, Limits::default(),
            Capacity::Capsule, native, Some((wave, prefix)), plan.input_bits().len(), false).unwrap();
        drop(q);
        assert_eq!(resources.work, seed.work + 6 * wave.construction_work
            + 2 * native.cumulative_work + 3 * native.template_copy_bytes);
        assert_eq!(resources.public_copy_bytes, seed.public_copy_bytes
            + 6 * wave.metadata_bytes + 3 * native.template_copy_bytes);
        assert_eq!(resources.kernels, seed.kernels + 1);
        drop(schedule);
        assert_eq!(budget.used_bytes(), before);
    }
    assert_eq!(budget.used_bytes(), 0);
}

fn small_quote(resources: &mut Resources, limits: Limits, cached: bool, wave: bool) -> Result<(), Error> {
    // Supplied-record control for the relocated prefix, already spent before
    // begin. These synthetic scalars are not a caller-facing admission API.
    if wave {
        resources.work += 6 * 11;
        resources.public_copy_bytes += 6 * quote_wave().metadata_bytes;
    }
    let mut q = resource_quote::Quote::begin(resources, limits, Capacity::Capsule,
        quote_native(), wave.then(|| (quote_wave(), 11)), 2, cached)?;
    // Runtime supplies reference/miss/retained/payload deltas 3/1/5/7 and
    // lowered metadata already priced across three roles: 3*7=21.
    if cached {
        q.tables(quote_stats(), 3, 1, 5, 7, 21)?;
    } else {
        // Uncached kernels may coexist with an existing retained cache, but
        // runtime requires NO new cache references (and hence no cache deltas).
        q.tables(quote_stats(), 0, 0, 0, 0, 0)?;
    }
    q.finish(quote_stats())
}

#[test]
fn quote_small_census_preserves_role_and_cache_multipliers() {
    // Small literal census, independent of Quote's arithmetic. Stats has 16
    // usize counters on this supported 64-bit target: three copies = 384 bytes.
    assert_eq!(std::mem::size_of::<cache::Stats>(), 128);
    // Cached lowering: 5*48 + 3*(4096+2*64)*3 + ceil(2049/2048)*192
    // =38640. Uncached lowering=2049*48=98352. Both add 64+8+1024.
    // Template copies=69; wave work=186 and copies=222. Table metadata copies
    // =72+512+21+384=989, payload copies=21; uncached snapshot copies=384.
    // Issuance/import=26 vs 13; no cache deltas on uncached kernels.
    // Native peaks: max(29+34,57)+300=363; wave adds 43 ->406.
    for (cached, wave, expected) in [
        (false, false, [3, 6, 99532, 14, 22, 479, 17, 19, 100, 363, 452]),
        (true, false, [3, 6, 39841, 14, 22, 1118, 17, 19, 100, 363, 452]),
        (false, true, [3, 6, 99718, 14, 93, 701, 17, 19, 100, 406, 495]),
        (true, true, [3, 6, 40027, 14, 93, 1340, 17, 19, 100, 406, 495]),
    ] {
        let mut resources = quote_seed();
        small_quote(&mut resources, Limits::default(), cached, wave).unwrap();
        assert_eq!(resource_fields(resources), expected, "cached={cached} wave={wave}");
    }
    // Previous peaks survive smaller later kernels. Construction peaks are not
    // interchangeable with combined-reservation or retained-cache envelopes.
    for (previous, construction, peak, combined) in [(900, 19, 900, 989), (29, 100, 600, 689)] {
        let mut resources = Resources { peak_native_bytes: previous, ..quote_seed() };
        let native = pllm_garble::fusefss_backend::composition::Resources {
            template_construction_bytes: construction, ..quote_native()
        };
        let mut q = resource_quote::Quote::begin(&mut resources, Limits::default(),
            Capacity::Capsule, native, None, 2, true).unwrap();
        q.tables(quote_stats(), 3, 1, 5, 7, 21).unwrap();
        q.finish(quote_stats()).unwrap();
        assert_eq!(resources.peak_native_bytes, peak);
        assert_eq!(resources.combined_reservation_bytes, combined);
    }
}

#[test]
fn quote_limits_accept_exact_reject_one_short() {
    for (cached, wave, work, frames, copies) in [
        (false, false, 99532, 22, 479), (true, false, 39841, 22, 1118),
        (false, true, 99718, 93, 701), (true, true, 40027, 93, 1340),
    ] {
        let exact = Limits { max_work: work, max_material_bytes: 14,
            max_frame_bytes: frames, max_public_copy_bytes: copies, ..Limits::default() };
        small_quote(&mut quote_seed(), exact, cached, wave).unwrap();
        for field in 0..4 {
            let mut short = exact;
            match field {
                0 => short.max_work -= 1,
                1 => short.max_material_bytes -= 1,
                2 => short.max_frame_bytes -= 1,
                _ => short.max_public_copy_bytes -= 1,
            }
            let mut resources = quote_seed();
            assert_eq!(small_quote(&mut resources, short, cached, wave), Err(Error::Resource));
            // These are final-phase refusals, not rollback to an earlier census.
            assert_eq!(resources.kernels, 3);
            assert_eq!(resources.work, work);
            assert_eq!(resources.public_copy_bytes, copies);
        }
    }
    for (capacity, limit) in [(Capacity::Capsule, 32768), (Capacity::Decoder, 1 << 22)] {
        for short in [false, true] {
            let mut resources = Resources { kernels: limit - usize::from(!short), ..quote_seed() };
            let mut q = resource_quote::Quote::begin(&mut resources, Limits::default(),
                capacity, quote_native(), None, 2, true).unwrap();
            q.tables(quote_stats(), 3, 1, 5, 7, 21).unwrap();
            assert_eq!(q.finish(quote_stats()), if short { Err(Error::Resource) } else { Ok(()) });
            assert_eq!(resources.kernels, limit + usize::from(short));
        }
    }
    let maximum = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES;
    for short in [false, true] {
        let mut resources = Resources { metadata_bytes: 0, live_array_bytes_per_party: 0, ..quote_seed() };
        // +2*17 Plan bytes and +3*(47+53) cache bytes exactly reach ceiling.
        let native = pllm_garble::fusefss_backend::composition::Resources {
            combined_reservation_bytes: maximum - 334 + usize::from(short), ..quote_native()
        };
        let mut q = resource_quote::Quote::begin(&mut resources, Limits::default(),
            Capacity::Capsule, native, None, 2, true).unwrap();
        q.tables(quote_stats(), 3, 1, 5, 7, 21).unwrap();
        assert_eq!(q.finish(quote_stats()), if short { Err(Error::Resource) } else { Ok(()) });
        assert_eq!(resources.combined_reservation_bytes, maximum + usize::from(short));
    }
}

#[test]
fn quote_overflow_returns_resource() {
    // All cases supply records directly: even impossible records must fail
    // checked arithmetic, not debug panic, release wrapping or saturation.
    for field in 0..9 {
        let mut resources = quote_seed();
        let mut native = quote_native();
        let mut wave = quote_wave();
        match field {
            0 => resources.kernels = usize::MAX,
            1 => resources.primitive_requests_per_party = usize::MAX,
            2 => resources.work = usize::MAX,
            3 => native.cumulative_work = usize::MAX / 2 + 1,
            4 => native.template_copy_bytes = usize::MAX / 3 + 1,
            5 => resources.public_copy_bytes = usize::MAX,
            6 => wave.construction_work = usize::MAX / 6 + 1,
            // Metadata copies are no longer charged by begin. A prefix larger
            // than total construction work must reject, never wrap residual.
            7 => wave.construction_work = 0,
            _ => native.secret_steps = usize::MAX,
        }
        assert!(matches!(resource_quote::Quote::begin(&mut resources, Limits::default(),
            Capacity::Capsule, native, Some((wave, usize::from(field == 7))), 2, true), Err(Error::Resource)), "begin {field}");
    }
    for field in 0..11 {
        let mut resources = quote_seed();
        let mut native = quote_native();
        let mut stats = quote_stats();
        let (mut inputs, mut references, mut misses, mut retained, mut copied, mut lowered) =
            (2, 3, 1, 5, 7, 21);
        match field {
            0 => inputs = usize::MAX / 32 + 1,
            1 => native.instructions = usize::MAX / 4 + 1,
            2 => stats.retained_entries = usize::MAX / 64 + 1,
            3 => references = usize::MAX,
            4 => misses = usize::MAX / 512 + 1,
            5 => retained = usize::MAX / 48 + 1,
            6 => copied = usize::MAX / 3 + 1,
            7 => lowered = usize::MAX,
            8 => native.public_table_reservation_bytes = usize::MAX,
            9 => stats.retained_entries = usize::MAX / 64,
            _ => references = usize::MAX / 4224,
        }
        let mut q = resource_quote::Quote::begin(&mut resources, Limits::default(),
            Capacity::Capsule, native, None, inputs, field != 8).unwrap();
        assert_eq!(q.tables(stats, references, misses, retained, copied, lowered),
            Err(Error::Resource), "tables {field}");
    }
    for field in 0..14 {
        let mut resources = quote_seed();
        let mut native = quote_native();
        let mut stats = quote_stats();
        let mut wave = quote_wave();
        match field {
            0 => resources.material_bytes = usize::MAX,
            1 => resources.frame_bytes = usize::MAX,
            2 => wave.frame_bytes_per_party = usize::MAX / 2 + 1,
            3 => native.public_table_copy_bytes_total = usize::MAX / 2 + 1,
            4 => native.combined_reservation_bytes = usize::MAX,
            5 => native.plan_reservation_bytes = usize::MAX / 2 + 1,
            6 => native.template_construction_bytes = usize::MAX / 3 + 1,
            7 => stats.retained_bytes = usize::MAX,
            8 => stats.retained_bytes = usize::MAX / 3 + 1,
            9 => resources.metadata_bytes = usize::MAX / 3 + 1,
            10 => resources.live_array_bytes_per_party = usize::MAX / 2 + 1,
            11 => resources.peak_native_bytes = usize::MAX,
            12 => native.private_material_bytes_total = usize::MAX,
            _ => stats.metadata_reservation_bytes = usize::MAX,
        }
        let mut q = resource_quote::Quote::begin(&mut resources, Limits::default(),
            Capacity::Capsule, native, Some((wave, 0)), 2, true).unwrap();
        q.tables(stats, 3, 1, 5, 7, 21).unwrap();
        assert_eq!(q.finish(stats), Err(Error::Resource), "finish {field}");
    }
    // Early public-copy refusal mutates the copied total but does not charge
    // template-copy work or proceed to any cache phase, preserving old ordering.
    let mut resources = quote_seed();
    let limits = Limits { max_public_copy_bytes: 81, ..Limits::default() };
    assert!(matches!(resource_quote::Quote::begin(&mut resources, limits,
        Capacity::Capsule, quote_native(), None, 2, true), Err(Error::Resource)));
    assert_eq!(resources.work, 15);
    assert_eq!(resources.public_copy_bytes, 82);
    assert_eq!(resources.kernels, 3);
    assert_eq!(resources.source_table_cache_bytes_per_role, 23);
}

// Specialized pre-extraction census for exactly I=1,O=2,P=2,V=2,width=2.
// Four padded-row one-hots and two argmaxes; Last once per epoch, one feedback
// gap. No cache references, lowered tables, templates, material or native driver.
// This does not call Quote, Schedule::run, or Schedule::execute as its oracle.
fn small_legacy_obligations(p: &Program) -> (Resources, [u8; 32], [u8; 32]) {
    let budget = request_budget(p).unwrap();
    let tables = request_tables(p, &budget).unwrap();
    let stats = tables.stats();
    let binding = kernels::Binding { source_digest: p.source.digest(), contract_digest: p.digest };
    let hot = kernels::token_one_hot(binding, Range::new(0, 1).unwrap().gate(), 2, &budget).unwrap().0;
    let maximum = if p.online == OnlineSchedule::WavefrontV1 {
        // Two leaves require exactly one stable-max lane, no further tree level.
        let plan = kernels::stable_max(binding, [p.nodes[1].range.gate(); 2],
            Range::new(0, 1).unwrap().gate(), &budget).unwrap().0;
        p.online.repeat(plan, 1).unwrap()
    } else {
        kernels::argmax(binding, &ranges(2, p.nodes[1].range).unwrap(), &budget).unwrap().0
    };
    let mut expected = Resources {
        metadata_bytes: p.resources.metadata_bytes,
        live_array_bytes_per_party: p.resources.live_array_bytes_per_party,
        source_table_cache_bytes_per_role: stats.metadata_reservation_bytes,
        public_copy_bytes: 3 * p.source.reservation_bytes(),
        work: 3 * p.resources.metadata_bytes + 3 * stats.metadata_reservation_bytes
            + 3 * stats.numeric_revision_hash_bytes,
        ..Resources::default()
    };
    let mut hash = Sha256::new();
    hash.update(b"pllm.fss-decoder.kernel-obligations.v1\0");
    hash.update(p.digest);
    hash.update(b"request-local-shared-source-tables.v1\0");
    hash.update(tables.numeric_revision());
    let mut refused = [0; 32];
    for selection in 0..2u64 {
        for (node, plan) in [(0u64, &hot), (0, &hot), (2, &maximum)] {
            let r = plan.resources();
            assert_eq!(r.public_table_reservation_bytes, 0);
            assert_eq!(r.public_table_copy_bytes_total, 0);
            assert_eq!(r.template_construction_bytes, 0);
            assert_eq!(r.template_copy_bytes, 0);
            let wave = if p.online == OnlineSchedule::WavefrontV1 {
                let wave = wavefront::Schedule::new(plan).unwrap();
                hash.update(wave.digest());
                Some(wave.resources())
            } else { None };
            // The unchanged scalar no-table formula, specialized to this tiny
            // schedule. Source-copy/geometry controls remain separate above.
            expected.kernels += 1;
            expected.primitive_requests_per_party += r.secret_steps;
            expected.work += 2 * r.cumulative_work + 32 * plan.input_bits().len()
                + 4 * r.instructions + 1024;
            expected.public_copy_bytes += 3 * std::mem::size_of::<cache::Stats>();
            expected.material_bytes += r.private_material_bytes_total;
            expected.frame_bytes += wave.map_or(r.frame_bytes_total, |w| 2 * w.frame_bytes_per_party);
            if let Some(wave) = wave {
                expected.work += 6 * wave.construction_work;
                expected.public_copy_bytes += 6 * wave.metadata_bytes;
            }
            let native = r.combined_reservation_bytes + 2 * r.plan_reservation_bytes
                + wave.map_or(0, |w| w.additional_reservation_bytes)
                + 3 * stats.metadata_reservation_bytes;
            expected.peak_native_bytes = expected.peak_native_bytes.max(native);
            // Exact old table-obligation order is 14 zero cumulative counters,
            // then selection, node, Plan. A final admission refusal commits the
            // table counters but MUST NOT commit that final invocation tuple.
            for _ in 0..14 { hash.update(0u64.to_le_bytes()); }
            if selection == 1 && node == 2 { refused = hash.clone().finalize().into(); }
            hash.update(selection.to_le_bytes());
            hash.update(node.to_le_bytes());
            hash.update(plan.digest());
        }
    }
    // Per epoch: embedding local=2*P*V*width=16, Last local=2*width=4.
    expected.work += 40;
    expected.combined_reservation_bytes = 3 * expected.metadata_bytes
        + 2 * expected.live_array_bytes_per_party + expected.peak_native_bytes;
    (expected, hash.finalize().into(), refused)
}

#[test]
fn quote_extraction_preserves_small_schedule_resources_and_obligations() {
    for online in [OnlineSchedule::Serial, OnlineSchedule::WavefrontV1] {
        let mut p = embedding(Capacity::Capsule, false, false, 2, 2, 1, 2);
        p.online = online;
        let (expected, obligations, refused) = small_legacy_obligations(&p);
        assert_eq!(expected.kernels, 6); // P, not growing prefixes 1+2.
        assert_ne!(obligations, refused);
        let exact = Limits {
            max_work: expected.work, max_material_bytes: expected.material_bytes,
            max_frame_bytes: expected.frame_bytes, max_public_copy_bytes: expected.public_copy_bytes,
            ..Limits::default()
        };
        p.limits = exact;
        for role in [0, 1] {
            for driven in [false, true] {
                let (actual, hash) = walk(&p, role, driven);
                assert_eq!(resource_fields(actual), resource_fields(expected));
                assert_eq!(hash, obligations);
            }
        }
        preflight(&mut p).unwrap();
        assert_eq!(resource_fields(p.resources), resource_fields(expected));
        assert_eq!(p.obligations, obligations);
        for field in 0..4 {
            p.limits = exact;
            match field {
                0 => p.limits.max_work -= 1,
                1 => p.limits.max_material_bytes -= 1,
                2 => p.limits.max_frame_bytes -= 1,
                _ => p.limits.max_public_copy_bytes -= 1,
            }
            for role in [0, 1] {
                for driven in [false, true] {
                    let budget = request_budget(&p).unwrap();
                    let tables = request_tables(&p, &budget).unwrap();
                    let mut driver = ZeroDriver::default();
                    let mut state = Schedule::new(&p, context(&p), role, None, &tables).unwrap();
                    if driven { state.driver = Some(&mut driver); }
                    assert!(matches!(state.execute(&[0]), Err(Error::Resource)));
                    assert_eq!(resource_fields(state.resources), resource_fields(expected));
                    assert_eq!(<[u8; 32]>::from(state.hash.clone().finalize()), refused);
                    drop(state);
                    assert_eq!(driver.calls, if driven { 5 } else { 0 });
                }
            }
        }
    }
}

#[test]
fn bootstrap_descriptor_matches_cache_stats() {
    // Retain the actual source/Program owner and ordinary admitted request cache.
    // Descriptor inspection itself neither reserves budget nor acquires tables.
    let p = embedding(Capacity::Capsule, false, false, 2, 2, 1, 1);
    let budget = request_budget(&p).unwrap();
    let limits = cache::Limits {
        max_entries: cache::MAX_ENTRIES,
        max_retained_bytes: cache::MAX_RETAINED_BYTES,
    };
    let descriptor = SourceTableCache::bootstrap_descriptor(p.digest, limits).unwrap();
    // Independent two-slot layout census; no production descriptor arithmetic
    // used as its oracle. Entry holds one aligned Identity and one Arc pointer.
    let slot = std::mem::size_of::<cache::Identity>() + std::mem::size_of::<usize>();
    for entries in [1, 2] {
        let small = SourceTableCache::bootstrap_descriptor(p.digest,
            cache::Limits { max_entries: entries, ..limits }).unwrap();
        assert_eq!(small.metadata_reservation_bytes,
            std::mem::size_of::<SourceTableCache<'_>>() + entries * slot + 4096);
    }
    assert_eq!(descriptor.numeric_revision_hash_bytes,
        include_bytes!("../../../../paper/reproduction-contracts/fusefss/gate-set/source-lock.json").len());
    assert_eq!(budget.used_bytes(), 0);
    let tables = request_tables(&p, &budget).unwrap();
    let stats = tables.stats();
    assert_eq!(stats, cache::Stats {
        metadata_reservation_bytes: descriptor.metadata_reservation_bytes,
        numeric_revision_hash_bytes: descriptor.numeric_revision_hash_bytes,
        ..cache::Stats::default()
    });
    assert_eq!(budget.used_bytes(), stats.metadata_reservation_bytes);
    assert!(stats.metadata_reservation_bytes >= 4096);
    assert!(stats.numeric_revision_hash_bytes > 0);
    assert_eq!(tables.bootstrap.work, 3 * p.resources.metadata_bytes
        + 3 * stats.metadata_reservation_bytes + 3 * stats.numeric_revision_hash_bytes);
    assert_eq!(tables.bootstrap.public_copy_bytes, 3 * p.source.reservation_bytes());
    let state = Schedule::new(&p, context(&p), 0, None, &tables).unwrap();
    assert_eq!(resource_fields(state.resources), resource_fields(tables.bootstrap));
    drop(state);
    drop(tables);
    assert_eq!(budget.used_bytes(), 0);

    // Shared validation retains Domain precedence even for enormous limits.
    for (namespace, entries, retained) in [
        ([0; 32], 1, 1), ([1; 32], 0, 1), ([1; 32], usize::MAX, 1),
        ([1; 32], 1, 0), ([1; 32], 1, usize::MAX),
    ] {
        let limits = cache::Limits { max_entries: entries, max_retained_bytes: retained };
        assert_eq!(SourceTableCache::bootstrap_descriptor(namespace, limits),
            Err(pllm_garble::fusefss_backend::Error::Domain));
        assert!(matches!(SourceTableCache::new(namespace, limits, &budget),
            Err(pllm_garble::fusefss_backend::Error::Domain)));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn bootstrap_exact_and_one_short() {
    // Independent small literal oracle: 3*7 + 3*13 + 3*17 = 111;
    // copies=3*19=57; compiler=3*7+2*11=43; floor=43+39=82.
    let descriptor = cache::BootstrapDescriptor {
        metadata_reservation_bytes: 13, numeric_revision_hash_bytes: 17,
    };
    for (work, copies, accepted) in [(111, 57, true), (110, 57, false), (111, 56, false)] {
        let mut factories = 0;
        let result = resource_quote::Quote::bootstrap(7, 11, 19, descriptor,
            Limits { max_work: work, max_public_copy_bytes: copies, ..Limits::default() },
            |resources| {
                // Same private boundary used immediately before real constructor.
                // Synthetic metadata only: no keys, tables, RNG, or placeholders.
                factories += 1;
                assert_eq!(resources.work, 111);
                assert_eq!(resources.public_copy_bytes, 57);
                assert_eq!(resources.source_table_cache_bytes_per_role, 13);
                assert_eq!(resources.metadata_bytes, 7);
                assert_eq!(resources.live_array_bytes_per_party, 11);
                assert_eq!(resources.peak_native_bytes, 0);
                assert_eq!(resources.combined_reservation_bytes, 0);
                Ok(())
            });
        assert_eq!(result, if accepted { Ok(()) } else { Err(Error::Resource) });
        assert_eq!(factories, usize::from(accepted));
    }

    let maximum = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES;
    // Exact floor: compiler=maximum-39, per-role available=13. Raise the
    // floor by exactly one byte using +3 cache bytes and -2 live bytes.
    let live = (maximum - 60) / 2;
    for short in [false, true] {
        let mut factories = 0;
        assert_eq!(resource_quote::Quote::available_per_role(7,
            live - usize::from(short)).unwrap(), 13);
        let result = resource_quote::Quote::bootstrap(7, live - usize::from(short), 19,
            cache::BootstrapDescriptor {
                metadata_reservation_bytes: 13 + usize::from(short),
                numeric_revision_hash_bytes: 17,
            }, Limits { max_work: usize::MAX, max_public_copy_bytes: usize::MAX,
                ..Limits::default() }, |_| { factories += 1; Ok(()) });
        assert_eq!(result, if short { Err(Error::Resource) } else { Ok(()) });
        assert_eq!(factories, usize::from(!short));
    }

    // Actual request_tables path, not the synthetic Quote closure above.
    // Independent layout/source-length oracle, already checked against ordinary
    // cache Stats in bootstrap_descriptor_matches_cache_stats.
    // Exact thresholds prove bootstrap admission to the constructor boundary,
    // NOT acceptance/construction of the whole Tables owner. Unsupported is the
    // deliberate probe sentinel: no real constructor/hash runs in these scopes.
    for meter in [0, 1, 2] {
        let mut p = embedding(Capacity::Capsule, false, false, 2, 2, 1, 1);
        let slot = std::mem::size_of::<cache::Identity>() + std::mem::size_of::<usize>();
        let metadata = std::mem::size_of::<SourceTableCache<'_>>()
            + cache::MAX_ENTRIES * slot + 4096;
        let revision = include_bytes!(
            "../../../../paper/reproduction-contracts/fusefss/gate-set/source-lock.json").len();
        p.limits.max_work = 3 * p.resources.metadata_bytes + 3 * metadata + 3 * revision;
        p.limits.max_public_copy_bytes = 3 * p.source.reservation_bytes();
        if meter == 1 { p.limits.max_work -= 1; }
        if meter == 2 { p.limits.max_public_copy_bytes -= 1; }
        let budget = request_budget(&p).unwrap();
        let (result, entries) = cache_constructor_probe::scoped(|| {
            request_tables(&p, &budget).map(drop)
        });
        assert_eq!(result, Err(if meter == 0 { Error::Unsupported } else { Error::Resource }));
        assert_eq!(entries, usize::from(meter == 0));
        assert_eq!(budget.used_bytes(), 0);
        assert_eq!(budget.admitted_peak_bytes(), 0);
    }
}

#[test]
fn bootstrap_overflow_precedes_factory() {
    let huge = usize::MAX;
    let maximum = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES;
    // (metadata, live, source reservation, cache metadata, revision bytes).
    // Each checked multiply/add and cap/available-budget refusal has its own
    // isolated case; factory trap must stay untouched for every refusal.
    for (metadata, live, source, cache, revision) in [
        (huge, 0, 0, 0, 0),             // 3*metadata
        (0, 0, 0, huge, 0),             // 3*cache
        (huge / 3, 0, 0, 1, 0),         // metadata + cache work
        (0, 0, 0, 0, huge),             // 3*revision
        (0, 0, 0, 1, huge / 3),         // revision + prior work
        (0, 0, huge, 0, 0),             // 3*source copies
        (0, huge, 0, 0, 0),             // 2*live
        (huge / 3 - 1, 2, 0, 0, 0),    // compiler addition
        (0, huge / 2, 0, 1, 0),         // compiler + 3*cache floor
        (0, maximum / 2 + 1, 0, 0, 0),  // available subtraction
        (0, maximum / 2, 0, 0, 0),      // no available bytes
        (1, (maximum - 4) / 2, 0, 0, 0), // available / 3 is zero
        (0, 0, 0, maximum / 3 + 1, 0),  // cache exceeds per-role cap
    ] {
        let mut factories = 0;
        let result = resource_quote::Quote::bootstrap(metadata, live, source,
            cache::BootstrapDescriptor {
                metadata_reservation_bytes: cache, numeric_revision_hash_bytes: revision,
            }, Limits { max_work: huge, max_public_copy_bytes: huge, ..Limits::default() },
            |_| { factories += 1; Ok(()) });
        assert_eq!(result, Err(Error::Resource),
            "metadata={metadata} live={live} source={source} cache={cache} revision={revision}");
        assert_eq!(factories, 0);
    }

    // Exercise the actual request_tables input arithmetic, using a real Budget
    // obtained before deliberately overflowing the retained metadata scalar.
    // This is a private-boundary robustness case, not a claim that public Program
    // construction admits such metadata: request_budget also rejects it earlier.
    // Existing FuseFSS checked-arithmetic contract returns Resource, not a new
    // Source/Overflow error or a fabricated overflowing source reservation.
    let mut p = embedding(Capacity::Capsule, false, false, 2, 2, 1, 1);
    let budget = request_budget(&p).unwrap();
    let original_metadata = p.resources.metadata_bytes;
    p.resources.metadata_bytes = usize::MAX;
    assert!(matches!(request_budget(&p), Err(Error::Resource)));
    let (result, entries) = cache_constructor_probe::scoped(|| {
        request_tables(&p, &budget).map(drop)
    });
    assert_eq!(result, Err(Error::Resource));
    assert_eq!(entries, 0);
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(budget.admitted_peak_bytes(), 0);

    // Shared constructor validation keeps Domain precedence over exhausted work
    // and copy caps; the actual constructor boundary must remain unentered.
    p.resources.metadata_bytes = original_metadata;
    p.digest = [0; 32];
    p.limits.max_work = 0;
    p.limits.max_public_copy_bytes = 0;
    let (result, entries) = cache_constructor_probe::scoped(|| {
        request_tables(&p, &budget).map(drop)
    });
    assert_eq!(result, Err(Error::Domain));
    assert_eq!(entries, 0);
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(budget.admitted_peak_bytes(), 0);
}
