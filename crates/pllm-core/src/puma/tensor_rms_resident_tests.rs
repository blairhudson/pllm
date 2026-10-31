//! AUTHOR-ONLY / ALL UNRUN under Native HOLD. Public integer interpretation is
//! not protected execution, allocator coverage, original-source or Arc evidence.
use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

const Q: i64 = 262_144;

fn args<'a>(gamma: &'a [i64], norm: bool) -> RmsArgs<'a> {
    RmsArgs { rows: 1, width: gamma.len(), input: Interval::new(-1, 1).unwrap(),
        gamma, input_norm: norm.then_some(gamma.len() as u128) }
}

fn source<'a>(a: &RmsArgs<'a>) -> ResidentFactorySource<'a> {
    match a.input_norm {
        None => Construction::RmsNormCertified { rows: a.rows, width: a.width, gamma_len: a.gamma.len() }
            .resident_parameters(FactoryParameters::RmsNormCertified { input: a.input, gamma: a.gamma }),
        Some(squared_norm) => Construction::RmsNormWithInputNorm { rows: a.rows, width: a.width, gamma_len: a.gamma.len() }
            .resident_parameters(FactoryParameters::RmsNormWithInputNorm { input: a.input, gamma: a.gamma, squared_norm }),
    }
}

fn legacy(a: &RmsArgs<'_>) -> (Program, u128) {
    match a.input_norm {
        None => Program::rms_norm_certified(a.rows, a.width, a.input, a.gamma),
        Some(bound) => Program::rms_norm_with_input_norm(a.rows, a.width, a.input, a.gamma, bound),
    }.unwrap()
}

fn parity(out: &FactoryOutput, expected: &(Program, u128)) {
    let (a, b) = (out.program(), &expected.0);
    assert_eq!(a.nodes, b.nodes);
    assert_eq!((a.len, a.lanes, a.numeric_profile), (b.len, b.lanes, b.numeric_profile));
    assert_eq!(a.digest, b.digest);
    assert_eq!(a.last_use, b.last_use);
    assert_eq!(a.slots, b.slots);
    assert_eq!(a.register_count, b.register_count);
    assert_eq!(a.bit_widths, b.bit_widths);
    assert_eq!(a.checked_validity, b.checked_validity);
    assert_eq!(a.public_affines, b.public_affines);
    assert_eq!(a.public_affine_count, b.public_affine_count);
    assert_eq!(a.work, b.work);
    assert_eq!(out.certificate(), ProducedCertificate::GeneralOutputRowSquared(expected.1));
}

// Independent floor-Q18 interpreter of public graph. No production arithmetic
// or numeric helper called. Bits are unsigned ring64; eager bounds checked.
fn floor_graph(p: &Program, input: &[i64]) -> Vec<i128> {
    let mut values: Vec<Vec<i128>> = Vec::new();
    for node in p.nodes[..p.len].iter().map(|n| n.unwrap()) {
        let get = |v: Value| &values[v.0 as usize];
        let out: Vec<i128> = match node.instruction {
            Instruction::Input => input.iter().copied().map(i128::from).collect(),
            Instruction::Affine { left, right, coefficients: c } => get(left).iter().enumerate()
                .map(|(i, x)| x * i128::from(c.left)
                    + right.map_or(0, |r| get(r)[i] * i128::from(c.right)) + i128::from(c.constant)).collect(),
            Instruction::Product { left, right, rescale, certified } => get(left).iter().zip(get(right))
                .map(|(x, y)| {
                    let raw = x * y;
                    if let Some(b) = certified {
                        assert!((i128::from(b.lower())..=i128::from(b.upper())).contains(&raw));
                    }
                    if rescale { raw.div_euclid(262_144) } else { raw }
                }).collect(),
            Instruction::Rescale { input, shift } => get(input).iter().map(|x| x.div_euclid(1_i128 << shift)).collect(),
            Instruction::Bits(input) => get(input).iter().map(|x| i128::from(*x as u64)).collect(),
            Instruction::LowBits(input, width) => get(input).iter().map(|x| x & ((1_i128 << width) - 1)).collect(),
            Instruction::FromBit(input) => get(input).clone(),
            Instruction::BitMap(input, map) => get(input).iter().map(|x| {
                let x = u64::try_from(*x).unwrap();
                i128::from(match map {
                    BitMap::LowBits(n) => x & (u64::MAX >> (64 - n)),
                    BitMap::ShiftRight(n) => x >> n,
                    BitMap::ShiftLeft(n) => x << n,
                    BitMap::ReverseLowBits(n) => (0..n).fold(0, |out, i| out | (((x >> i) & 1) << (n - 1 - i))),
                    BitMap::FoldPairs => (0..32).fold(0, |out, i| {
                        let pair = ((x >> (2 * i)) ^ (x >> (2 * i + 1))) & 1;
                        out | (pair << i)
                    }),
                    BitMap::Parity(mask) => u64::from((x & mask).count_ones() % 2),
                })
            }).collect(),
            Instruction::BitOr(a, b) | Instruction::BitOrLow(a, b, _) => {
                let mask = match node.instruction { Instruction::BitOrLow(_, _, n) => (1_i128 << n) - 1, _ => -1 };
                get(a).iter().zip(get(b)).map(|(x, y)| (x | y) & mask).collect()
            }
            Instruction::BitXor(a, b) => get(a).iter().zip(get(b)).map(|(x, y)| x ^ y).collect(),
            Instruction::RowMap(input, width, RowMap::Sum) => get(input).chunks_exact(width).map(|row| row.iter().sum()).collect(),
            Instruction::RowMap(input, width, RowMap::Broadcast) => get(input).iter()
                .flat_map(|x| std::iter::repeat(*x).take(width)).collect(),
            Instruction::LaneAffine(input, table) => get(input).iter().zip(&p.public_affines[table as usize])
                .map(|(x, c)| { assert_eq!(c.right, 0); x * i128::from(c.left) + i128::from(c.constant) }).collect(),
            other => panic!("unexpected RMS instruction: {other:?}"),
        };
        assert_eq!(out.len(), node.lanes);
        if !node.bit {
            for x in &out {
                assert!((i128::from(node.bounds.lower())..=i128::from(node.bounds.upper())).contains(x), "{node:?}: {x}");
            }
        }
        values.push(out);
    }
    values.pop().unwrap()
}

// Separately specified literal equations, no graph or production constants.
// Cases below have width 2 or 896: both inverse-width coefficients are exact
// literals from the prescribed float32-to-Q18 rule (131072 and 292).
fn floor_equations(row: &[i64], gamma: &[i64]) -> Vec<i128> {
    let tr = |x: i128| x.div_euclid(262_144);
    let coefficient = match row.len() { 2 => 131_072, 896 => 292, _ => panic!("oracle geometry") };
    let sum: i128 = row.iter().map(|&x| tr(i128::from(x) * i128::from(x))).sum();
    let d = tr(sum * coefficient) + 1;
    assert!((1..(1_i128 << 35)).contains(&d));
    let mut j = 0;
    while (1_i128 << (j + 1)) <= d { j += 1; }
    let u = tr(d * (1_i128 << (34 - j)));
    let u2 = tr(u * u);
    let u3 = tr(u2 * u);
    let u4 = tr(u3 * u);
    let polynomial = 1_086_023 + tr(-4_057_974 * u + 10_085_068 * u2 - 13_072_088 * u3 + 6_823_457 * u4);
    let compensation = (if j % 2 == 1 { 724 } else { 512 }) * (1_i128 << (17 - (j + 1) / 2));
    let inverse = tr(polynomial * compensation);
    row.iter().zip(gamma).map(|(&x, &g)| tr(tr(i128::from(x) * inverse) * i128::from(g))).collect()
}

#[test]
fn both_variants_literal_small_integer_q18_and_zero_certificates_width_896_signed_gamma() {
    for norm in [false, true] {
        for width in [2, 896] {
            let gamma: Vec<_> = (0..width).map(|i| [Q, -Q, 0][i % 3]).collect();
            let a = args(&gamma, norm);
            let expected = legacy(&a);
            let mut slot = RmsResidentSlot::empty();
            let mut out = None;
            slot.build_into(&a, &mut out).unwrap();
            let out = out.as_ref().unwrap();
            parity(out, &expected);
            assert_eq!(GAMMA.with(|g| g.get()), (gamma.as_ptr(), gamma.len()));
            assert_eq!(out.program().numeric_profile(), RMS_NORM_PROFILE);
            let row: Vec<_> = (0..width).map(|i| [1, -1, 0][i % 3]).collect();
            let actual = floor_graph(out.program(), &row);
            assert_eq!(actual, floor_equations(&row, &gamma));
            // d=1, u=65536, polynomial=524249, inverse=134207744.
            // floor(+inverse/Q)=511, floor(-inverse/Q)=-512.
            assert_eq!(&actual[..2], &[511, 512]);
            assert_eq!(floor_graph(out.program(), &vec![0; width]), vec![0; width]);

            let mut zero_input = args(&gamma, norm);
            zero_input.input = Interval::new(0, 0).unwrap();
            zero_input.input_norm = norm.then_some(0);
            let mut zero_slot = RmsResidentSlot::empty();
            let mut zero_out = None;
            zero_slot.build_into(&zero_input, &mut zero_out).unwrap();
            // E=0: normalized radius=ceil_sqrt(width); max |gamma|=Q;
            // final radius twice that. Literal certificates, not measured output.
            assert_eq!(zero_out.as_ref().unwrap().certificate(),
                ProducedCertificate::GeneralOutputRowSquared(if width == 2 { 16 } else { 3600 }));

            let gamma_zero = vec![0; width];
            let mut zero_gamma_slot = RmsResidentSlot::empty();
            let mut zero_gamma_out = None;
            zero_gamma_slot.build_into(&args(&gamma_zero, norm), &mut zero_gamma_out).unwrap();
            assert_eq!(zero_gamma_out.as_ref().unwrap().certificate(),
                ProducedCertificate::GeneralOutputRowSquared(if width == 2 { 4 } else { 900 }));
            assert_eq!(floor_graph(zero_gamma_out.as_ref().unwrap().program(), &row), vec![0; width]);
        }
    }
}

fn arm(stage: RmsStage, invocation: usize, panic: bool) {
    assert!(invocation > 0);
    FAULT.with(|fault| assert!(fault.replace(Some((stage, invocation, panic))).is_none()));
}

fn resident_program(slot: &RmsResidentSlot) -> &Program {
    slot.program.as_ref().unwrap_or_else(|| &slot.builder.as_ref().unwrap().builder.as_ref().unwrap().0)
}

fn watch(pointer: *const Affine, capacity: usize) {
    assert!(!pointer.is_null());
    BACKING.with(|b| b.set((pointer, capacity, 0)));
}
fn arm_backing() {
    unwatch();
    OBSERVE_NEXT.with(|next| assert!(!next.replace(true)));
}
fn assert_backing(pointer: *const Affine, capacity: usize) {
    assert_eq!(BACKING.with(|b| { let (p, c, _) = b.get(); (p, c) }), (pointer, capacity));
    assert!(!OBSERVE_NEXT.with(|next| next.get()));
}
fn drops() -> usize { BACKING.with(|b| b.get().2) }
fn unwatch() {
    BACKING.with(|b| b.set((std::ptr::null(), 0, 0)));
    SECOND_BACKING.with(|b| b.set((std::ptr::null(), 0, 0)));
}

#[test]
fn every_safe_boundary_err_and_unwind_poison_whole_slot_and_keep_real_backing_until_teardown() {
    for norm in [false, true] {
        for stage in [RmsStage::Installed, RmsStage::Recipe, RmsStage::PostAffine, RmsStage::PostRecipe, RmsStage::PostTransfer] {
            for panic in [false, true] {
                let gamma = [Q, -Q];
                let a = args(&gamma, norm);
                let mut slot = RmsResidentSlot::empty();
                let mut out = None;
                if matches!(stage, RmsStage::PostAffine | RmsStage::PostRecipe | RmsStage::PostTransfer) {
                    arm_backing();
                }
                arm(stage, 1, panic);
                // Owners live OUTSIDE catch frame; closure borrows them.
                let result = catch_unwind(AssertUnwindSafe(|| slot.build_into(&a, &mut out)));
                if panic { assert!(result.is_err()); } else { assert_eq!(result.unwrap(), Err(Error::Bounds)); }
                assert_eq!(slot.state, State::Failed);
                assert!(out.is_none());
                if matches!(stage, RmsStage::Installed | RmsStage::Recipe) {
                    assert_eq!(resident_program(&slot).public_affine_count, 0);
                    assert_eq!(slot.builder.as_ref().unwrap().state, ResidentState::Ready);
                } else {
                    let p = resident_program(&slot);
                    assert_eq!(p.public_affine_count, 1);
                    assert_eq!(p.public_affines[0].len(), 2);
                    assert_eq!(GAMMA.with(|g| g.get()), (gamma.as_ptr(), 2));
                    assert_backing(p.public_affines[0].as_ptr(), p.public_affines[0].capacity());
                    assert_eq!(drops(), 0);
                }
                let digest = resident_program(&slot).digest;
                assert_eq!(slot.build_into(&a, &mut out), Err(Error::Phase));
                assert_eq!(resident_program(&slot).digest, digest);
                if matches!(stage, RmsStage::PostAffine | RmsStage::PostRecipe | RmsStage::PostTransfer) {
                    assert_eq!(drops(), 0);
                    drop(slot);
                    assert_eq!(drops(), 1);
                    unwatch();
                }
            }
        }
    }
}

#[test]
fn real_pending_reserve_and_partial_finish_unwinds_retain_original_backing() {
    for norm in [false, true] {
        for reserve in [true, false] {
            let gamma = [Q, 0];
            let a = args(&gamma, norm);
            let mut slot = RmsResidentSlot::empty();
            let mut out = None;
            arm_backing();
            if reserve { resident_builder::arm_affine_reserve_panic(); }
            else { resident_builder::arm_finish_panic(1); }
            let counts = resident_builder::finish_hook_counts();
            assert!(catch_unwind(AssertUnwindSafe(|| slot.build_into(&a, &mut out))).is_err());
            assert_eq!(slot.state, State::Failed);
            let resident = slot.builder.as_ref().unwrap();
            assert_eq!(resident.state, ResidentState::Failed);
            let (pointer, capacity) = if reserve {
                let observed = resident_builder::reserved_affine();
                assert_eq!((resident.pending_affine.as_ptr(), resident.pending_affine.capacity()), observed);
                assert!(observed.1 >= 2);
                assert!(resident.pending_affine.is_empty());
                assert_eq!(resident_program(&slot).public_affine_count, 0);
                observed
            } else {
                assert_eq!(resident_builder::finish_hook_counts(), (counts.0 + 1, counts.1 + 1));
                assert!(resident_program(&slot).register_count > 0);
                assert_eq!(resident_program(&slot).public_affine_count, 1);
                let table = &resident_program(&slot).public_affines[0];
                (table.as_ptr(), table.capacity())
            };
            assert_backing(pointer, capacity);
            assert!(out.is_none());
            assert_eq!(slot.build_into(&a, &mut out), Err(Error::Phase));
            assert_eq!(drops(), 0);
            drop(slot);
            assert_eq!(drops(), 1);
            unwatch();
        }
    }
}

#[test]
fn transferred_factory_output_owns_table_after_schedule_teardown_and_commit_refusals_keep_program() {
    for norm in [false, true] {
        let gamma = [Q, -Q];
        let a = args(&gamma, norm);
        let mut slot = RmsResidentSlot::empty();
        let mut out = None;
        arm_backing();
        slot.build_into(&a, &mut out).unwrap();
        assert_eq!(slot.state, State::Complete);
        assert_eq!(slot.builder.as_ref().unwrap().state, ResidentState::Transferred);
        let table = &out.as_ref().unwrap().program().public_affines[0];
        assert_backing(table.as_ptr(), table.capacity());
        drop(slot);
        assert_eq!(drops(), 0);
        drop(out);
        assert_eq!(drops(), 1);
        unwatch();

        let (p, c) = legacy(&a);
        let mut retained = Some(p);
        let pointer = retained.as_ref().unwrap().public_affines[0].as_ptr();
        let mut occupied = None;
        let mut another = RmsResidentSlot::empty();
        another.build_into(&a, &mut occupied).unwrap();
        let occupied_digest = occupied.as_ref().unwrap().program().digest;
        assert_eq!(work_preflight::commit_rms_output(&mut retained, c, &mut occupied), Err(Error::Binding));
        assert_eq!(retained.as_ref().unwrap().public_affines[0].as_ptr(), pointer);
        assert_eq!(occupied.as_ref().unwrap().program().digest, occupied_digest);
        let mut empty_out = None;
        assert_eq!(work_preflight::commit_rms_output(&mut None, c, &mut empty_out), Err(Error::Phase));
        assert!(empty_out.is_none());
        watch(pointer, retained.as_ref().unwrap().public_affines[0].capacity());
        work_preflight::commit_rms_output(&mut retained, c, &mut empty_out).unwrap();
        assert!(retained.is_none());
        assert_eq!(empty_out.as_ref().unwrap().program().public_affines[0].as_ptr(), pointer);
        assert_eq!(drops(), 0);
        drop(empty_out);
        assert_eq!(drops(), 1);
        unwatch();
    }
}

fn priced<'a>(a: &RmsArgs<'a>, tail: bool, workspace: &mut WorkPreflightWorkspace) -> ResidentNativeSchedule<'a> {
    let mut full_source = Some(source(a));
    let mut tail_source = tail.then(|| source(a));
    let mut schedule = WorkPreflightWorkspace::resident_schedule();
    schedule.install(&mut full_source, tail.then_some(&mut tail_source), 3).unwrap();
    assert!(full_source.is_none() && tail_source.is_none());
    schedule.bind().unwrap();
    schedule.preflight(workspace).unwrap();
    schedule
}

#[test]
fn admitted_full_then_tail_failure_never_refunds_or_revives_and_build_hook_precedes_factory_counter() {
    for norm in [false, true] {
        for panic in [false, true] {
            let gamma = [Q, 0];
            let a = args(&gamma, norm);
            let mut workspace = WorkPreflightWorkspace::empty();
            let mut schedule = priced(&a, true, &mut workspace);
            let unit = legacy(&a).0.work_budget();
            let expected_work = unit.checked_add(unit).unwrap().checked_add(unit).unwrap().checked_add(unit).unwrap();
            assert_eq!(schedule.summary().unwrap().work, expected_work);
            let mut total = WorkBudget::local();
            let prior = total;
            schedule.admit(&mut CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap()).unwrap();
            assert_eq!(total, prior.checked_add(expected_work).unwrap());
            let charged = total;
            let attempts = rms_work::attempts();
            let mut full = None;
            let mut tail = None;
            arm_backing();
            OBSERVE_SECOND.with(|next| assert!(!next.replace(true)));
            arm(RmsStage::PostAffine, 2, panic);
            let result = catch_unwind(AssertUnwindSafe(|| schedule.build_into(&mut full, &mut tail)));
            if panic { assert!(result.is_err()); } else { assert_eq!(result.unwrap(), Err(Error::Bounds)); }
            assert_eq!(rms_work::attempts(), [attempts[0] + 2, attempts[1], attempts[2]]);
            assert!(full.is_some() && tail.is_none());
            let full_digest = full.as_ref().unwrap().program().digest;
            let table = &full.as_ref().unwrap().program().public_affines[0];
            assert_backing(table.as_ptr(), table.capacity());
            let (tail_pointer, tail_capacity, tail_drops) = SECOND_BACKING.with(|b| b.get());
            assert!(!tail_pointer.is_null() && tail_pointer != table.as_ptr());
            assert_eq!(tail_capacity, 2);
            assert_eq!(tail_drops, 0);
            assert!(!OBSERVE_SECOND.with(|next| next.get()));
            workspace.reset();
            assert_eq!(schedule.build_into(&mut full, &mut tail), Err(Error::Phase));
            assert!(schedule.bind().is_err());
            assert!(schedule.preflight(&mut workspace).is_err());
            assert!(schedule.admit(&mut CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap()).is_err());
            assert_eq!(full.as_ref().unwrap().program().digest, full_digest);
            assert_eq!(total, charged);
            assert_eq!(SECOND_BACKING.with(|b| b.get().2), 0);
            drop(schedule);
            assert_eq!(drops(), 0);
            assert_eq!(SECOND_BACKING.with(|b| b.get().2), 1);
            drop(full);
            assert_eq!(drops(), 1);
            assert_eq!(SECOND_BACKING.with(|b| b.get().2), 1);
            unwatch();
        }

        let gamma = [Q, 0];
        let a = args(&gamma, norm);
        let mut workspace = WorkPreflightWorkspace::empty();
        let mut schedule = priced(&a, false, &mut workspace);
        let mut total = WorkBudget::local();
        schedule.admit(&mut CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap()).unwrap();
        let charged = total;
        let attempts = rms_work::attempts();
        let mut full = None;
        let mut tail = None;
        work_preflight::arm_resident_build_panic(1);
        assert!(catch_unwind(AssertUnwindSafe(|| schedule.build_into(&mut full, &mut tail))).is_err());
        assert_eq!(rms_work::attempts(), attempts); // Build hook BEFORE entered(0).
        assert_eq!(total, charged);
        assert_eq!(schedule.build_into(&mut full, &mut tail), Err(Error::Phase));
    }
}

#[test]
fn admitted_full_tail_success_keeps_row_order_exact_work_certificates_and_one_shot_permission() {
    for norm in [false, true] {
        let full_gamma = [Q, -Q];
        let tail_gamma = [0, Q];
        let mut full_args = args(&full_gamma, norm);
        full_args.rows = 2;
        let tail_args = args(&tail_gamma, norm);
        let full_expected = legacy(&full_args);
        let tail_expected = legacy(&tail_args);
        let work = full_expected.0.work_budget().checked_mul(3).unwrap()
            .checked_add(tail_expected.0.work_budget()).unwrap();
        let mut full_source = Some(source(&full_args));
        let mut tail_source = Some(source(&tail_args));
        let mut schedule = WorkPreflightWorkspace::resident_schedule();
        let mut workspace = WorkPreflightWorkspace::empty();
        schedule.install(&mut full_source, Some(&mut tail_source), 3).unwrap();
        schedule.bind().unwrap();
        schedule.preflight(&mut workspace).unwrap();
        assert_eq!(schedule.summary().unwrap().work, work);
        let mut total = WorkBudget::local();
        let prior = total;
        schedule.admit(&mut CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap()).unwrap();
        let attempts = rms_work::attempts();
        let rows = work_preflight::row_entries();
        let mut full = None;
        let mut tail = None;
        schedule.build_into(&mut full, &mut tail).unwrap();
        assert_eq!(rms_work::attempts(), [attempts[0] + 2, attempts[1], attempts[2]]);
        assert_eq!(work_preflight::row_entries(), rows + 2);
        assert_eq!(total, prior.checked_add(work).unwrap());
        parity(full.as_ref().unwrap(), &full_expected);
        parity(tail.as_ref().unwrap(), &tail_expected);
        assert_eq!(GAMMA.with(|g| g.get()), (tail_gamma.as_ptr(), 2));
        assert_eq!(floor_graph(full.as_ref().unwrap().program(), &[1, -1, 0, 1]), vec![511, 512, 0, -511]);
        assert_eq!(floor_graph(tail.as_ref().unwrap().program(), &[1, -1]), vec![0, -512]);
        workspace.reset();
        assert_eq!(schedule.build_into(&mut full, &mut tail), Err(Error::Phase));
        assert_eq!(total, prior.checked_add(work).unwrap());
    }
}

#[test]
fn real_geometry_binding_work_and_occupied_output_refusals_preserve_error_order_and_burn() {
    for norm in [false, true] {
        let gamma = [Q, 0];
        let mut a = args(&gamma, norm);
        a.rows = 0;
        a.width = 0;
        let mut slot = RmsResidentSlot::empty();
        let mut out = None;
        let rows = work_preflight::row_entries();
        let attempts = rms_work::attempts();
        assert_eq!(slot.build_into(&a, &mut out), Err(Error::Shape));
        assert_eq!(work_preflight::row_entries(), rows);
        assert_eq!(rms_work::attempts(), [attempts[0] + 1, attempts[1], attempts[2]]);
        assert_eq!(slot.state, State::Failed);
        assert!(slot.builder.is_none());
        let a = args(&gamma, norm);
        assert_eq!(slot.build_into(&a, &mut out), Err(Error::Phase));

        let construction = if norm {
            Construction::RmsNormWithInputNorm { rows: 1, width: 2, gamma_len: 2 }
        } else {
            Construction::RmsNormCertified { rows: 1, width: 2, gamma_len: 2 }
        };
        let mut mismatch = Some(construction
            .resident_parameters(FactoryParameters::Negative { input: a.input }));
        let mut schedule = WorkPreflightWorkspace::resident_schedule();
        schedule.install(&mut mismatch, None, 1).unwrap();
        let attempts = rms_work::attempts();
        assert!(matches!(schedule.bind(), Err(WorkPreflightError::ParameterBinding)));
        assert_eq!(rms_work::attempts(), attempts);

        let mut workspace = WorkPreflightWorkspace::empty();
        let mut schedule = priced(&a, false, &mut workspace);
        let mut total = WorkBudget::local();
        let prior = total;
        // Exact existing-work cap at current total: new request cannot fit.
        let mut short = CallerWorkLedger::new(&mut total, prior.operations(), prior.cursors().aes_work).unwrap();
        assert!(matches!(schedule.admit(&mut short), Err(WorkPreflightError::Native(Error::Bounds))));
        assert_eq!(short.total(), prior);
        drop(short);
        assert_eq!(total, prior);
        assert_eq!(rms_work::attempts(), attempts);
        assert_eq!(schedule.build_into(&mut out, &mut None), Err(Error::Phase));

        let mut schedule = priced(&a, false, &mut workspace);
        schedule.admit(&mut CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap()).unwrap();
        let charged = total;
        let mut output_slot = RmsResidentSlot::empty();
        output_slot.build_into(&a, &mut out).unwrap();
        let digest = out.as_ref().unwrap().program().digest;
        let pointer = out.as_ref().unwrap().program().public_affines[0].as_ptr();
        let attempts = rms_work::attempts();
        assert_eq!(schedule.build_into(&mut out, &mut None), Err(Error::Binding));
        assert_eq!(rms_work::attempts(), attempts);
        assert_eq!(total, charged);
        assert_eq!(out.as_ref().unwrap().program().digest, digest);
        assert_eq!(out.as_ref().unwrap().program().public_affines[0].as_ptr(), pointer);
        out = None;
        workspace.reset();
        assert_eq!(schedule.build_into(&mut out, &mut None), Err(Error::Phase));
        assert!(out.is_none());
        assert_ne!(digest, [0; 32]);
    }
}

#[test]
fn proposal_inventory_names_inline_returns_but_never_grants_native_prepaid_authority() {
    let gamma = [Q, 0];
    for norm in [false, true] {
        let quote = RmsResidentSlot::quote(&args(&gamma, norm)).unwrap();
        assert_eq!(quote.slot_bytes, std::mem::size_of::<RmsResidentSlot>());
        assert_eq!(quote.schedule_bytes, std::mem::size_of::<ResidentNativeSchedule<'_>>());
        assert_eq!(quote.row_program_bytes, std::mem::size_of::<Program>());
        assert_eq!(quote.row_builder_bytes, std::mem::size_of::<Builder>());
        assert_eq!(quote.row_result_bytes, std::mem::size_of::<Result<Builder, Error>>());
        assert_eq!(quote.resident_argument_bytes, std::mem::size_of::<Builder>());
        assert_eq!(quote.args_bytes, std::mem::size_of::<RmsArgs<'_>>());
        assert_eq!(quote.sink_bytes, std::mem::size_of::<RmsResidentSink<'_>>());
        assert_eq!(quote.builder_option_bytes, std::mem::size_of::<Option<ResidentBuilder>>());
        assert_eq!(quote.recipe_option_bytes, std::mem::size_of::<Option<(Value, u128)>>());
        assert_eq!(quote.recipe_result_bytes, std::mem::size_of::<Result<(Value, u128), Error>>());
        assert_eq!(quote.program_option_bytes, std::mem::size_of::<Option<Program>>());
        assert_eq!(quote.output_option_bytes, std::mem::size_of::<Option<FactoryOutput>>());
        assert_eq!(quote.output_result_bytes, std::mem::size_of::<Result<FactoryOutput, Error>>());
        assert_eq!(quote.unit_result_bytes, std::mem::size_of::<Result<(), Error>>());
        assert_eq!(quote.legacy.public_table_bytes, 2 * std::mem::size_of::<Affine>());
        assert!(quote.legacy.proof_buffer_bytes > 0 && quote.legacy.finish_buffer_bytes > 0);
    }
}
