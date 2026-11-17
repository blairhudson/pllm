use super::rms_work::attempts;

fn old_xor(b: &mut Builder, input: Value, constant: u64) -> Result<Value, Error> {
    b.bit_width(input)?;
    b.push(
        Instruction::BitConstantXor(input, constant),
        Interval::new(0, 0)?,
        true,
    )
}
fn old_outside(b: &mut Builder, bits: Value, bound: i64, below: bool) -> Result<Value, Error> {
    let key = (bound as u64) ^ (1 << 63);
    let ordered = old_xor(b, bits, 1 << 63)?;
    let mut spread = old_xor(b, ordered, key)?;
    for shift in [1, 2, 4, 8, 16, 32] {
        let lower = b.bit_map(spread, BitMap::ShiftRight(shift))?;
        spread = b.bit_or(spread, lower)?;
    }
    let lower = b.bit_map(spread, BitMap::ShiftRight(1))?;
    let highest = b.bit_xor(spread, lower)?;
    b.bit_map(highest, BitMap::Parity(if below { key } else { !key }))
}
// Literal pre-extraction codec; no new RmsSink/checked_recipe/all_valid_recipe.
fn old_checked(
    lanes: usize,
    incoming: Interval,
    accepted: Interval,
    fallback: i64,
) -> Result<Program, Error> {
    let mut b = Builder::new(lanes, incoming)?;
    let input = b.input();
    b.node(input, false)?;
    if fallback < accepted.lower() || fallback > accepted.upper() {
        return Err(Error::Bounds);
    }
    b.0.numeric_profile = CHECKED_RANGE_PROFILE;
    let bits = b.bits(input)?;
    let below = old_outside(&mut b, bits, accepted.lower(), true)?;
    let above = old_outside(&mut b, bits, accepted.upper(), false)?;
    let invalid = b.bit_or(below, above)?;
    let valid_bit = old_xor(&mut b, invalid, 1)?;
    let valid = b.from_bit(valid_bit)?;
    let mut mask = valid_bit;
    for shift in [1, 2, 4, 8, 16, 32] {
        let upper = b.bit_map(mask, BitMap::ShiftLeft(shift))?;
        mask = b.bit_xor(mask, upper)?;
    }
    let delta = old_xor(&mut b, bits, fallback as u64)?;
    let union = b.bit_or(mask, delta)?;
    let xor = b.bit_xor(mask, delta)?;
    let intersection = b.bit_xor(union, xor)?;
    let selected = old_xor(&mut b, intersection, fallback as u64)?;
    let low = b.bit_map(selected, BitMap::LowBits(32))?;
    let low = b.from_boolean(low)?;
    let high = b.bit_map(selected, BitMap::ShiftRight(32))?;
    let high = b.bit_map(high, BitMap::LowBits(31))?;
    let high = b.from_boolean(high)?;
    let sign = b.bit_map(selected, BitMap::ShiftRight(63))?;
    let sign = b.from_bit(sign)?;
    let negative = b.affine(
        sign,
        None,
        Affine {
            left: i64::MIN,
            right: 0,
            constant: 0,
        },
    )?;
    let upper = b.affine(
        high,
        Some(negative),
        Affine {
            left: 1 << 32,
            right: 1,
            constant: 0,
        },
    )?;
    let safe = b.affine(
        upper,
        Some(low),
        Affine {
            left: 1,
            right: 1,
            constant: 0,
        },
    )?;
    b.0.nodes[safe.0 as usize].as_mut().unwrap().bounds = accepted;
    b.finish_checked(safe, valid)
}

#[test]
fn checked_native_original_codec_and_all_valid_vectors_profiles_and_hashes_unchanged() {
    for lanes in [1, 3, 128] {
        for (lo, hi, fallback) in [
            (-32, 32, 0),
            (i64::MIN, i64::MAX, i64::MIN),
            (i64::MIN, i64::MAX, i64::MAX),
        ] {
            let input = Interval::new(i64::MIN, i64::MAX).unwrap();
            let accepted = Interval::new(lo, hi).unwrap();
            let a = Program::checked_range(lanes, input, accepted, fallback).unwrap();
            let e = old_checked(lanes, input, accepted, fallback).unwrap();
            assert_eq!(a.nodes, e.nodes);
            assert_eq!(a.digest(), e.digest());
            assert_eq!(a.bit_widths, e.bit_widths);
            assert_eq!(a.checked_validity, e.checked_validity);
            assert_eq!(a.work_budget(), e.work_budget());
            let mut b = Builder::row(lanes, Interval::new(0, 1).unwrap()).unwrap();
            b.0.numeric_profile = CHECKED_RANGE_PROFILE;
            let sum = b.row_map(b.input(), lanes, RowMap::Sum).unwrap();
            let d = b
                .affine(
                    sum,
                    None,
                    Affine {
                        left: 1,
                        right: 0,
                        constant: -(lanes as i64),
                    },
                )
                .unwrap();
            let failed = b.sign(d).unwrap();
            let failed = b.from_bit(failed).unwrap();
            let v = b
                .affine(
                    failed,
                    None,
                    Affine {
                        left: -1,
                        right: 0,
                        constant: 1,
                    },
                )
                .unwrap();
            let e = b.finish(v).unwrap();
            let a = Program::all_valid(lanes).unwrap();
            assert_eq!(a.nodes, e.nodes);
            assert_eq!(a.digest(), e.digest());
            assert_eq!(a.work_budget(), e.work_budget());
        }
    }
}
use super::*;

#[test]
fn checked_rows_geometry_query_error_order_and_poisoning() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    let accepted = Interval::new(-1, 1).unwrap();
    for (rows, width, error) in [
        (0, usize::MAX, Error::Shape), (usize::MAX, 0, Error::Shape),
        (usize::MAX, 2, Error::Bounds), (9, 1, Error::Bounds),
        (1, 129, Error::Bounds), (8, 128, Error::Bounds), (5, 128, Error::Bounds),
    ] {
        let mut space = RmsWorkWorkspace::empty();
        let before = attempts();
        assert_eq!(Program::checked_range_rows_work(rows, width, full, accepted, 9, &mut space), Err(error));
        assert_eq!(attempts(), before);
        assert!(space.is_poisoned());
        assert!(matches!(Program::checked_range_rows(rows, width, full, accepted, 9), Err(e) if e == error));
        assert_eq!(Program::checked_range_rows_work(1, 1, full, accepted, 0, &mut space), Err(Error::Phase));
        space.reset();
        assert!(Program::checked_range_rows_work(1, 1, full, accepted, 0, &mut space).is_ok());
    }
    assert!(matches!(Program::checked_range_rows(4, 128, full, accepted, 9), Err(Error::Bounds)));
    // The public tile entry's original <=128 restriction is not relaxed.
    assert!(matches!(Program::checked_range(129, full, accepted, 0), Err(Error::Bounds)));
    assert!(matches!(Builder::new(129, full), Err(Error::Bounds)));
}

#[test]
fn checked_rows_query_native_independent_instruction_work_census() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    for (rows, width) in [(1, 1), (3, 7), (1, 128), (3, 128), (4, 128), (8, 64)] {
        let lanes = rows * width;
        let mut space = RmsWorkWorkspace::empty();
        let before = attempts();
        let query = Program::checked_range_rows_work(rows, width, full, full, 0, &mut space).unwrap();
        assert_eq!(attempts(), before); // no native constructor entry
        let program = Program::checked_range_rows(rows, width, full, full, 0).unwrap();
        space.assert_native_metadata(&program);
        let mut counts = [0usize; 8];
        let mut work = WorkBudget::default();
        for node in program.nodes[..program.len].iter().flatten() {
            let part = match node.instruction {
                Instruction::Input => { counts[0] += 1; WorkBudget::default() }
                Instruction::Bits(_) => { counts[1] += 1; WorkBudget::conversion(lanes, false).unwrap() }
                Instruction::BitMap(..) => { counts[2] += 1; WorkBudget::local() }
                Instruction::BitConstantXor(..) => { counts[3] += 1; WorkBudget::local() }
                Instruction::BitOr(a, b) => { counts[4] += 1; WorkBudget::bit_or(lanes,
                    program.bit_widths[a.0 as usize].max(program.bit_widths[b.0 as usize])).unwrap() }
                Instruction::BitXor(..) => { counts[5] += 1; WorkBudget::local() }
                Instruction::FromBit(v) => { counts[6] += 1;
                    WorkBudget::from_boolean(lanes, program.bit_widths[v.0 as usize]).unwrap() }
                Instruction::Affine { .. } => { counts[7] += 1; WorkBudget::local() }
                _ => panic!("unexpected checked Rows recipe instruction"),
            };
            work = work.checked_add(part).unwrap();
        }
        assert_eq!(counts, [1, 1, 26, 7, 14, 10, 4, 3]);
        assert_eq!(program.len, 66);
        assert_eq!(query, work);
        assert_eq!(program.work_budget(), work);
        assert_eq!(program.validity_lanes(), Some(lanes));
        assert_eq!(program.output_lanes(), lanes);
        let reduce = Program::all_valid(lanes).unwrap();
        assert_eq!(reduce.output_lanes(), 1);
        assert_eq!(reduce.instructions(), 5);
        assert_eq!(Program::all_valid_work(lanes, &mut space).unwrap(), reduce.work_budget());
    }
}

// Same ordinary replicated-input construction as the existing native drivers.
// Full-word test shares deliberately do not restrict incoming private values.
fn checked_rows_inputs(runtimes: &[PartyRuntime; 3], values: &[i64]) -> [ReplicatedShare; 3] {
    checked_rows_try_inputs(runtimes, values).unwrap()
}

fn checked_rows_try_inputs(runtimes: &[PartyRuntime; 3], values: &[i64]) -> Result<[ReplicatedShare; 3], Error> {
    let op = Operation::new(runtimes[0].context(), 0, Party::P0, values.len(), 18, Limits::default())?;
    let a: Vec<_> = (0..values.len()).map(|i| 0x8765432112345678u64.wrapping_add(i as u64)).collect();
    let b: Vec<_> = (0..values.len()).map(|i| 0xfedcba98ffffffffu64.wrapping_sub(i as u64)).collect();
    let c: Vec<_> = values.iter().enumerate().map(|(i, &x)| (x as u64).wrapping_sub(a[i]).wrapping_sub(b[i])).collect();
    let parts = [&a[..], &b[..], &c[..]];
    let [a, b, c] = [Party::P0, Party::P1, Party::P2].map(|p| ReplicatedShare::from_local(op, p,
        parts[p as usize], parts[p.next() as usize], Limits::default()));
    Ok([a?, b?, c?])
}

fn checked_rows_clear(values: [&ReplicatedShare; 3]) -> Vec<i64> {
    for p in 0..3 { assert_eq!(values[p].second(), values[(p + 1) % 3].first()); }
    (0..values[0].first().len()).map(|i| values.iter().fold(0u64,
        |acc, value| acc.wrapping_add(value.first()[i])) as i64).collect()
}

// Public delivery barriers, no new numeric protocol. Ordinary encoded delivery
// copies each outbound byte string then drops its Encoded<Frame> BEFORE any
// receiver call, matching the consuming Python binding boundary. Borrowing
// as_bytes() alone would keep issuer leases and is NOT normal-drop cadence.
// The separate retained-history mode keeps REAL original owners until drop.
fn checked_rows_delivery(program: &Program, values: &[i64], encoded: bool, retain: bool, shared: bool)
    -> Result<(Vec<i64>, Vec<i64>, usize), Error> {
    let runtimes = crate::puma::multiply::tests::parties(program.work_budget().limits().unwrap());
    checked_rows_delivery_with_runtimes(program, values, encoded, retain, shared, runtimes, &mut [false; 3], None)
}

// Fixed public-boundary diagnostics only. No frames, source state, observer Arc,
// event history, failed-allocation demand or operation-local peak is retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CheckedRowsStage {
    Bind, Preflight, WorkLedger, WorkAdmission, NativeBuild, Resources,
    RuntimeLimits, RuntimeSetup, InputSetup, Executor, Start, EncodeCopy,
    EncodedReceive, FinishStep, FinishChecked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CheckedRowsRawFailure { Native(Error), Work(WorkPreflightError) }
impl From<Error> for CheckedRowsRawFailure {
    fn from(error: Error) -> Self { Self::Native(error) }
}
impl From<WorkPreflightError> for CheckedRowsRawFailure {
    fn from(error: WorkPreflightError) -> Self { Self::Work(error) }
}

#[derive(Clone, Copy, Debug, Default)]
struct CheckedRowsUsage {
    live: usize,
    cumulative_successful_reservation_peak: usize,
    limit: usize,
}
fn checked_rows_usage(usages: &[PayloadUsage; 3]) -> [CheckedRowsUsage; 3] {
    usages.each_ref().map(|u| CheckedRowsUsage { live: u.live_bytes(),
        cumulative_successful_reservation_peak: u.peak_bytes(), limit: u.limit_bytes() })
}

#[derive(Clone, Copy, Debug)]
struct CheckedRowsCall {
    stage: CheckedRowsStage,
    ordinal: usize,
    party: Option<usize>,
    step: Option<usize>,
    instruction: Option<std::mem::Discriminant<Instruction>>,
    lanes: usize,
    public_bit_width: Option<u8>,
    wide_from_bit: bool,
    round: usize,
    sender: Option<Party>,
    recipient: Option<Party>,
    phase: Option<MessagePhase>,
    before: [CheckedRowsUsage; 3],
    after: [CheckedRowsUsage; 3],
    cursor_before: Option<crate::puma::prss::Cursors>,
    cursor_after: Option<crate::puma::prss::Cursors>,
    cursors_unchanged: Option<bool>,
    complete_before: Option<bool>,
    retired_before: Option<bool>,
    complete_after: Option<bool>,
    retired_after: Option<bool>,
    input_credit: Option<usize>,
    reconstructed_admission_sum: Option<usize>,
    // These are RECONSTRUCTED, never observed inside start_step. None means
    // no applicable wide FromBit, or copies cannot both fit: planner unresolved.
    reconstructed_planner_entry_live: Option<usize>,
    reconstructed_planner_sum: Option<usize>,
    retained: usize,
    largest_encoded_frame: usize,
    succeeded: Option<bool>,
}

#[derive(Clone, Copy, Debug)]
struct CheckedRowsFirstFailure { raw: CheckedRowsRawFailure, call: CheckedRowsCall }

#[derive(Debug)]
struct CheckedRowsDiagnostic {
    rows: usize,
    width: usize,
    work: Option<WorkBudget>,
    factory_summary: Option<FactoryWorkSummary>,
    resources: Option<Resources>,
    ordinal: usize,
    current: Option<CheckedRowsCall>,
    first_failure: Option<CheckedRowsFirstFailure>,
    admissions: [Option<CheckedRowsCall>; 3],
    // Exactly two wide FromBit instructions, three ordinary parties. None is
    // NOT REACHED, never a simulated/replayed later PPA entry.
    wide_from_bit: [Option<CheckedRowsCall>; 6],
    wide_from_bit_count: usize,
    cleanup: [CheckedRowsUsage; 3],
    largest_encoded_frame: usize,
    completed: [bool; 3],
}
impl CheckedRowsDiagnostic {
    fn new(rows: usize, width: usize) -> Self {
        Self { rows, width, work: None, factory_summary: None, resources: None,
            ordinal: 0, current: None, first_failure: None, admissions: [None; 3],
            wide_from_bit: [None; 6], wide_from_bit_count: 0,
            cleanup: [CheckedRowsUsage::default(); 3], largest_encoded_frame: 0,
            completed: [false; 3] }
    }
    fn begin(&mut self, stage: CheckedRowsStage) {
        assert!(self.first_failure.is_none());
        self.ordinal += 1;
        self.current = Some(CheckedRowsCall { stage, ordinal: self.ordinal,
            party: None, step: None, instruction: None, lanes: self.rows * self.width,
            public_bit_width: None, wide_from_bit: false,
            round: 0, sender: None, recipient: None, phase: None,
            before: [CheckedRowsUsage::default(); 3], after: [CheckedRowsUsage::default(); 3],
            cursor_before: None, cursor_after: None, cursors_unchanged: None,
            complete_before: None, retired_before: None,
            complete_after: None, retired_after: None, input_credit: None,
            reconstructed_admission_sum: None, reconstructed_planner_entry_live: None,
            reconstructed_planner_sum: None, retained: 0, largest_encoded_frame: 0, succeeded: None });
    }
    fn before_owner(&mut self, stage: CheckedRowsStage, usages: &[PayloadUsage; 3],
        owner: &Executor, round: usize, message: Option<(Party, Party, MessagePhase)>,
        retained: usize, largest: usize) {
        self.begin(stage);
        let call = self.current.as_mut().unwrap();
        call.before = checked_rows_usage(usages);
        call.party = Some(owner.party() as usize);
        call.step = Some(owner.step);
        call.cursor_before = Some(owner.cursors());
        call.complete_before = Some(owner.complete());
        call.retired_before = Some(owner.retired());
        call.round = round;
        call.retained = retained;
        call.largest_encoded_frame = largest;
        if let Some((sender, recipient, phase)) = message {
            call.sender = Some(sender); call.recipient = Some(recipient); call.phase = Some(phase);
        }
        if let Some(node) = owner.program.nodes.get(owner.step).and_then(Option::as_ref) {
            call.instruction = Some(std::mem::discriminant(&node.instruction));
            call.lanes = node.lanes;
            call.public_bit_width = Some(owner.program.bit_widths[owner.step]);
            if let Instruction::FromBit(input) = node.instruction {
                let bits = owner.program.bit_widths[input.0 as usize];
                call.public_bit_width = Some(bits);
                call.wide_from_bit = bits > 8;
                if stage == CheckedRowsStage::Start && bits > 8 {
                    let live = call.before[owner.party() as usize].live;
                    let copied = node.lanes.checked_mul(16).and_then(|n| live.checked_add(n));
                    if let Some(entry) = copied.filter(|&n| n <= call.before[owner.party() as usize].limit) {
                        call.reconstructed_planner_entry_live = Some(entry);
                        call.reconstructed_planner_sum = node.lanes.checked_mul(320).and_then(|n| entry.checked_add(n));
                    }
                }
            }
        }
    }
    fn observe<T, E: Copy + Into<CheckedRowsRawFailure>>(&mut self, result: &Result<T, E>) {
        let call = self.current.as_mut().unwrap();
        call.succeeded = Some(result.is_ok());
        call.cursors_unchanged = call.cursor_before.zip(call.cursor_after).map(|(a, b)| a == b);
        if let Err(error) = result {
            assert!(self.first_failure.is_none());
            self.first_failure = Some(CheckedRowsFirstFailure { raw: (*error).into(), call: *call });
        }
        if call.stage == CheckedRowsStage::Executor {
            self.admissions[call.party.unwrap()] = Some(*call);
        }
        if call.stage == CheckedRowsStage::Start && call.wide_from_bit {
            assert!(self.wide_from_bit_count < self.wide_from_bit.len());
            self.wide_from_bit[self.wide_from_bit_count] = Some(*call);
            self.wide_from_bit_count += 1;
        }
    }
    fn after_owner<T>(&mut self, usages: &[PayloadUsage; 3], owner: &Executor,
        largest: usize, result: &Result<T, Error>) {
        let call = self.current.as_mut().unwrap();
        call.after = checked_rows_usage(usages);
        call.cursor_after = Some(owner.cursors());
        call.complete_after = Some(owner.complete());
        call.retired_after = Some(owner.retired());
        call.largest_encoded_frame = largest;
        self.largest_encoded_frame = largest;
        self.observe(result);
    }
    fn cleaned(&mut self, usages: &[PayloadUsage; 3]) {
        self.cleanup = checked_rows_usage(usages);
        for u in self.cleanup {
            assert_eq!(u.live, 0);
            assert_eq!(u.limit, 258_048);
            assert!(u.cumulative_successful_reservation_peak <= u.limit);
        }
    }
}

fn checked_rows_copy_outbound(frame: Frame, retain: bool,
    retained: &mut Vec<crate::puma::wire::Encoded<Frame>>, largest: &mut usize,
    ot_helpers: &mut [bool; 3]) -> Result<(usize, Party, zeroize::Zeroizing<Vec<u8>>), Error> {
    let to = frame.recipient() as usize;
    let from = frame.sender();
    if frame.inner.header().phase == MessagePhase::BooleanOtMask {
        assert_eq!(frame.recipient(), from.next());
        ot_helpers[from as usize] = true;
    }
    let wire = frame.encode()?;
    assert_eq!(&wire.as_bytes()[..4], b"PUT1");
    *largest = (*largest).max(wire.as_bytes().len());
    // Binding cadence: encoded bytes coexist with this real independent copy;
    // no borrowed byte slice, Frame clone, or alias survives in normal mode.
    let raw = zeroize::Zeroizing::new(wire.as_bytes().to_vec());
    if retain { retained.push(wire); } else { drop(wire); }
    Ok((to, from, raw))
}

fn checked_rows_delivery_with_runtimes(program: &Program, values: &[i64], encoded: bool,
    retain: bool, shared: bool, runtimes: [PartyRuntime; 3], ot_helpers: &mut [bool; 3],
    mut diagnostic: Option<&mut CheckedRowsDiagnostic>)
    -> Result<(Vec<i64>, Vec<i64>, usize), Error> {
    let usages = runtimes.each_ref().map(PartyRuntime::payload_usage);
    if let Some(d) = diagnostic.as_deref_mut() {
        assert!(encoded && !retain && !shared);
        assert!(program.validity_lanes().is_some());
        for u in &usages { assert_eq!(u.limit_bytes(), 258_048); }
        d.begin(CheckedRowsStage::InputSetup);
        d.current.as_mut().unwrap().before = checked_rows_usage(&usages);
    }
    let input_result = if diagnostic.is_some() { checked_rows_try_inputs(&runtimes, values) }
        else { Ok(checked_rows_inputs(&runtimes, values)) };
    if let Some(d) = diagnostic.as_deref_mut() {
        d.current.as_mut().unwrap().after = checked_rows_usage(&usages);
        d.observe(&input_result);
    }
    let input_values = match input_result {
        Ok(inputs) => inputs,
        Err(error) => {
            drop(runtimes);
            assert!(usages.iter().all(|u| u.live_bytes() == 0));
            if let Some(d) = diagnostic.as_deref_mut() { d.cleaned(&usages); }
            return Err(error);
        }
    };
    let mut inputs = input_values.map(Some).into_iter();
    let template = Arc::new(program.clone());
    let mut owners = Vec::new();
    let mut runtime_iter = runtimes.into_iter();
    while let Some(runtime) = runtime_iter.next() {
        let party = runtime.party() as usize;
        let input = inputs.next().unwrap().unwrap();
        if let Some(d) = diagnostic.as_deref_mut() {
            d.begin(CheckedRowsStage::Executor);
            let resources = d.resources.unwrap();
            let credit = input.first.charged_bytes(&usages[party])
                .checked_add(input.second.charged_bytes(&usages[party])).unwrap();
            let call = d.current.as_mut().unwrap();
            call.party = Some(party);
            call.cursor_before = Some(runtime.cursors());
            call.before = checked_rows_usage(&usages);
            call.input_credit = Some(credit);
            call.reconstructed_admission_sum = Some(call.before[party].live.checked_sub(credit).unwrap()
                .checked_add(resources.owned_admitted_bytes).unwrap().checked_sub(4096).unwrap());
        }
        let owner = if shared { Executor::new_shared(runtime, template.clone(), input) }
            else { Executor::new(runtime, program.clone(), input) };
        if let Some(d) = diagnostic.as_deref_mut() {
            let call = d.current.as_mut().unwrap();
            call.after = checked_rows_usage(&usages);
            if let Ok(owner) = &owner {
                call.cursor_after = Some(owner.cursors());
                call.complete_after = Some(owner.complete());
                call.retired_after = Some(owner.retired());
            }
            d.observe(&owner);
        }
        match owner { Ok(owner) => owners.push(owner), Err(error) => {
            drop((owners, inputs, template));
            // Explicitly drop the same unconsumed iterator before checking cleanup.
            drop(runtime_iter);
            assert!(usages.iter().all(|u| u.live_bytes() == 0));
            if let Some(d) = diagnostic.as_deref_mut() { d.cleaned(&usages); }
            return Err(error);
        } }
    }
    let mut retained = Vec::new();
    let mut retained_typed = Vec::new();
    let mut largest = 0;
    let mut round = 0;
    macro_rules! observed {
        ($stage:expr, $owner:expr, $message:expr, $call:expr) => {{
            if let Some(d) = diagnostic.as_deref_mut() {
                d.before_owner($stage, &usages, &$owner, round, $message,
                    retained.len() + retained_typed.len(), largest);
            }
            let result = $call;
            if let Some(d) = diagnostic.as_deref_mut() {
                d.after_owner(&usages, &$owner, largest, &result);
            }
            result
        }};
    }
    let run = (|| -> Result<(), Error> {
        for _ in 0..program.instructions() {
            round = 0;
            let mut frames = Vec::new();
            let mut copied_batch = Vec::new();
            for owner in &mut owners {
                for frame in observed!(CheckedRowsStage::Start, owner, None, owner.start_step())?.into_iter().flatten() {
                    if encoded {
                        let message = (frame.sender(), frame.recipient(), frame.inner.header().phase);
                        let copied = observed!(CheckedRowsStage::EncodeCopy, owner, Some(message),
                            checked_rows_copy_outbound(frame, retain, &mut retained, &mut largest, ot_helpers))?;
                        copied_batch.push((copied, message));
                    } else { frames.push(frame); }
                }
            }
            while !frames.is_empty() || !copied_batch.is_empty() {
                round += 1;
                let mut next = Vec::new();
                let mut next_copied = Vec::new();
                for frame in frames {
                    let to = frame.recipient() as usize;
                    let from = frame.sender();
                    if frame.inner.header().phase == MessagePhase::BooleanOtMask {
                        // Existing OT endpoints: helper -> receiver; sender is
                        // receiver.next(). Observing every helper exercises every
                        // party in all three public roles, without exposing bits.
                        assert_eq!(frame.recipient(), from.next());
                        ot_helpers[from as usize] = true;
                    }
                        if retain {
                            // Retain actual original typed Frame owners. A real
                            // decoded delivery gets the recipient's ordinary outer
                            // metadata reservation, as receive_encoded does.
                            let inner = Message::decode(&frame.inner.encode_bytes()?, from, frame.recipient())?;
                            let delivered = Frame { program: frame.program, invocation: frame.invocation,
                                step: frame.step, inner, compact_bits: None,
                                _metadata: owners[to].payload_usage().reserve(std::mem::size_of::<Frame>())? };
                            retained_typed.push(frame);
                            next.extend(owners[to].receive(delivered)?.into_iter().flatten());
                        } else { next.extend(owners[to].receive(frame)?.into_iter().flatten()); }
                }
                // Every original in the consumed outbound batch has now dropped
                // (unless this is the explicit negative retained-history mode).
                // Delivery itself is never treated as lease retirement.
                for ((to, from, raw), message) in copied_batch {
                    // Consume/copy/drop EACH returned native batch before the
                    // next peer call, not merely before the next delivery round.
                    for frame in observed!(CheckedRowsStage::EncodedReceive, owners[to], Some(message),
                        owners[to].receive_encoded(&raw, from))?.into_iter().flatten() {
                        let message = (frame.sender(), frame.recipient(), frame.inner.header().phase);
                        let copied = observed!(CheckedRowsStage::EncodeCopy, owners[to], Some(message),
                            checked_rows_copy_outbound(frame, retain, &mut retained, &mut largest, ot_helpers))?;
                        next_copied.push((copied, message));
                    }
                }
                frames = next;
                copied_batch = next_copied;
            }
            for owner in &mut owners {
                observed!(CheckedRowsStage::FinishStep, owner, None, owner.finish_step())?;
            }
        }
        Ok(())
    })();
    if let Err(error) = run {
        drop((owners, retained, retained_typed, template, inputs));
        assert!(usages.iter().all(|u| u.live_bytes() == 0));
        if let Some(d) = diagnostic.as_deref_mut() { d.cleaned(&usages); }
        return Err(error);
    }
    if program.validity_lanes().is_none() {
        // Preserve ordinary callers' original terminal assertions/panic behavior.
        let results: Vec<_> = owners.into_iter().map(|owner| owner.finish().unwrap()).collect();
        let output = checked_rows_clear([&results[0].1, &results[1].1, &results[2].1]);
        drop((results, retained, retained_typed, template, inputs));
        assert!(usages.iter().all(|u| u.live_bytes() == 0));
        return Ok((output, Vec::new(), largest));
    }
    let results: Vec<_> = if diagnostic.is_none() {
        owners.into_iter().map(|owner| owner.finish_checked().unwrap()).collect()
    } else {
        let mut results = Vec::new();
        let mut remaining = owners.into_iter();
        while let Some(owner) = remaining.next() {
            let party = owner.party() as usize;
            let d = diagnostic.as_deref_mut().unwrap();
            d.before_owner(CheckedRowsStage::FinishChecked, &usages, &owner, round, None, 0, largest);
            d.completed[party] = owner.complete();
            let result = owner.finish_checked();
            let call = d.current.as_mut().unwrap();
            call.after = checked_rows_usage(&usages);
            if let Ok((runtime, _, _)) = &result { call.cursor_after = Some(runtime.cursors()); }
            d.observe(&result);
            match result {
                Ok(result) => results.push(result),
                Err(error) => {
                    drop((remaining, results, retained, retained_typed, template, inputs));
                    assert!(usages.iter().all(|u| u.live_bytes() == 0));
                    d.cleaned(&usages);
                    return Err(error);
                }
            }
        }
        results
    };
    let safe = checked_rows_clear([&results[0].1, &results[1].1, &results[2].1]);
    let valid = checked_rows_clear([&results[0].2, &results[1].2, &results[2].2]);
    drop((results, retained, retained_typed, template, inputs));
    assert!(usages.iter().all(|u| u.live_bytes() == 0));
    if let Some(d) = diagnostic.as_deref_mut() { d.cleaned(&usages); }
    Ok((safe, valid, largest))
}

#[test]
fn checked_rows_numeric_full_i64_common_interval_and_fallback_oracle() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    for (lo, hi, fallback) in [(-7, 11, 3), (1, 1, 1),
        (i64::MIN, -1, i64::MIN), (0, i64::MAX, i64::MAX),
        (i64::MIN, i64::MAX, 0), (i64::MIN, i64::MIN, i64::MIN),
        (i64::MAX, i64::MAX, i64::MAX)] {
        let boundary = [i64::MIN, i64::MIN + 1, i64::MAX - 1, i64::MAX,
            -1, 0, 1, lo, hi, lo.saturating_sub(1), hi.saturating_add(1), fallback];
        let accepted = Interval::new(lo, hi).unwrap();
        let program = Program::checked_range_rows(3, 4, full, accepted, fallback).unwrap();
        for (encoded, shared) in [(false, false), (true, false), (true, true)] {
            let (safe, valid, _) = checked_rows_delivery(&program, &boundary, encoded, false, shared).unwrap();
            for (i, &x) in boundary.iter().enumerate() {
                let ok = i128::from(x) >= i128::from(lo) && i128::from(x) <= i128::from(hi);
                assert_eq!(valid[i], i64::from(ok));
                assert_eq!(safe[i], if ok { x } else { fallback });
            }
            assert_eq!(valid.iter().all(|&v| v == 1), boundary.iter().all(|&x| x >= lo && x <= hi));
        }
    }
}

// Strict success for these two genuine ordinary requests at unchanged caps.
// ND-INNER-OBSERVATION still limits any future refusal diagnosis: boundary
// snapshots cannot identify a nested refusal or measure operation-local peak.
fn checked_rows_factory_normal_drop_fits_existing_caps(rows: usize, expected_largest: usize) {
    assert_eq!(MAX_ADMITTED_BYTES, 262_144);
    assert_eq!(FIXED_RESERVE_BYTES, 4096);
    assert_eq!(MAX_LANES, 1024);
    let width = 128;
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    let values: Vec<_> = (0..rows * width).map(|i| [i64::MIN, 0, i64::MAX][i % 3]).collect();
    let mut diagnostic = CheckedRowsDiagnostic::new(rows, width);
    let result = (|| -> Result<(Vec<i64>, Vec<i64>, usize), CheckedRowsRawFailure> {
        macro_rules! stage {
            ($stage:expr, $call:expr) => {{
                diagnostic.begin($stage);
                let result = $call;
                diagnostic.observe(&result);
                result.map_err(CheckedRowsRawFailure::from)?
            }};
        }
        let factory = stage!(CheckedRowsStage::Bind,
            Construction::CheckedRangeRows { rows, width }.bind(
                FactoryParameters::CheckedRangeRows { incoming: full, accepted: full, fallback: 0 }));
        let full_factory = {
            // Scope preparation workspace out before native construction/execution.
            let mut workspace = WorkPreflightWorkspace::empty();
            let ticket = stage!(CheckedRowsStage::Preflight, factory.preflight(1, &mut workspace));
            let summary = ticket.summary();
            diagnostic.work = Some(summary.work);
            diagnostic.factory_summary = Some(summary.full);
            assert!(summary.tail.is_none());
            assert_eq!(summary.full.invocations, 1);
            // Existing numeric TEST-fixture sponsor convention, not SOURCE money.
            let mut total = WorkBudget::default();
            let mut ledger = stage!(CheckedRowsStage::WorkLedger,
                CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX));
            let admitted = stage!(CheckedRowsStage::WorkAdmission, ledger.admit(ticket));
            assert_eq!(ledger.total(), summary.work);
            let (full, tail) = admitted.into_parts();
            assert!(tail.is_none());
            full
        };
        let output = stage!(CheckedRowsStage::NativeBuild, full_factory.build());
        assert_eq!(output.certificate(), ProducedCertificate::None);
        let program = output.program();
        assert_eq!(Some(program.work_budget()), diagnostic.work);
        let resources = stage!(CheckedRowsStage::Resources, program.resources());
        diagnostic.resources = Some(resources);
        // Independent actual-resource formula; invocation already includes registers.
        assert_eq!(resources.runtime_fixed_reserve_bytes, 4096);
        let common = 4096usize.checked_add(resources.invocation_container_bytes).unwrap()
            .checked_add(resources.frame_bytes).unwrap().checked_add(resources.working_bytes).unwrap();
        assert_eq!(resources.owned_admitted_bytes, common.checked_add(resources.public_table_bytes).unwrap());
        assert_eq!(resources.shared_admitted_bytes, common.checked_add(resources.immutable_template_bytes).unwrap());
        let limits = stage!(CheckedRowsStage::RuntimeLimits, program.work_budget().limits());
        diagnostic.begin(CheckedRowsStage::RuntimeSetup);
        // Existing parties fixture has no fallible return; its own setup assertions
        // remain assertions, not fabricated native/work errors.
        let runtimes = crate::puma::multiply::tests::parties(limits);
        diagnostic.observe(&Ok::<(), Error>(()));
        checked_rows_delivery_with_runtimes(program, &values, true, false, false,
            runtimes, &mut [false; 3], Some(&mut diagnostic)).map_err(CheckedRowsRawFailure::from)
    })();
    // All native owners, results, queues, inputs and template have dropped before
    // this one concise terminal summary. Full first-failure call appears only
    // on refusal. Peaks are cumulative successful reservations, not operation peaks.
    let max_reconstructed_ppa_sum = diagnostic.wide_from_bit.iter().flatten()
        .filter_map(|call| call.reconstructed_planner_sum).max();
    eprintln!("checked_rows_factory_normal_drop: geometry={}x{} terminal={:?} resources={:?} cumulative_successful_reservation_peaks={:?} widest_frame={} complete={:?} live={:?} wide_count={} max_RECONSTRUCTED_PPA_sum={:?} first_failure={:?}",
        rows, width, result.as_ref().map(|_| ()).map_err(|error| *error), diagnostic.resources,
        diagnostic.cleanup.map(|u| u.cumulative_successful_reservation_peak),
        diagnostic.largest_encoded_frame, diagnostic.completed, diagnostic.cleanup.map(|u| u.live),
        diagnostic.wide_from_bit_count, max_reconstructed_ppa_sum, diagnostic.first_failure);
    let (safe, valid, largest) = result.unwrap_or_else(|raw| panic!(
        "strict normal-drop success required: raw={raw:?}, first_failure={:?}", diagnostic.first_failure));
    assert!(diagnostic.first_failure.is_none());
    assert!(diagnostic.resources.is_some(), "factory/resource entry must succeed");
    assert_eq!(diagnostic.cleanup.map(|u| u.live), [0; 3]);
    assert_eq!(diagnostic.cleanup.map(|u| u.limit), [258_048; 3]);
    for admission in diagnostic.admissions {
        let admission = admission.expect("all three Executor admissions reached");
        let party = admission.party.unwrap();
        assert_eq!(admission.succeeded, Some(true));
        assert!(admission.reconstructed_admission_sum.unwrap() <= admission.before[party].limit);
    }
    assert_eq!(diagnostic.completed, [true; 3]);
    assert_eq!(diagnostic.wide_from_bit_count, 6);
    for call in diagnostic.wide_from_bit {
        let call = call.expect("all six wide FromBit starts reached");
        assert_eq!(call.succeeded, Some(true));
        assert_eq!(call.retained, 0);
        let limit = call.before[call.party.unwrap()].limit;
        assert!(call.reconstructed_planner_entry_live.unwrap() <= limit);
        assert!(call.reconstructed_planner_sum.unwrap() <= limit);
    }
    assert_eq!(safe.len(), rows * width);
    assert_eq!(valid.len(), rows * width);
    for (i, &input) in values.iter().enumerate() {
        let accepted = i128::from(input) >= i128::from(i64::MIN)
            && i128::from(input) <= i128::from(i64::MAX);
        assert!(valid[i] == 0 || valid[i] == 1);
        assert_eq!(valid[i], i64::from(accepted));
        assert_eq!(safe[i], if accepted { input } else { 0 });
        assert_eq!(valid[i], 1);
        assert_eq!(safe[i], input);
    }
    assert_eq!(largest, 16 * values.len() + 92);
    assert_eq!(largest, expected_largest);
    assert_eq!(diagnostic.largest_encoded_frame, largest);
}

#[test]
fn checked_rows_factory_normal_drop_384_fits_existing_caps() {
    checked_rows_factory_normal_drop_fits_existing_caps(3, 6236);
}

#[test]
fn checked_rows_factory_normal_drop_512_fits_existing_caps() {
    checked_rows_factory_normal_drop_fits_existing_caps(4, 8284);
}

#[test]
fn checked_rows_real_admission_and_encoded_retained_frames_do_not_raise_caps() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    for (rows, width) in [(1, 32), (3, 128), (4, 128)] {
        let p = Program::checked_range_rows(rows, width, full, full, 0).unwrap();
        let values: Vec<_> = (0..rows * width).map(|i| [i64::MIN, 0, i64::MAX][i % 3]).collect();
        for shared in [false, true] {
            for encoded in [false, true] {
                // Independent normal-drop candidate: a real Bounds outcome is
                // honest, but retained-history obstruction cannot establish it.
                // Both wide PPA entries/current registers and intra-operation
                // peaks still need native observation, separate from admission.
                let result = checked_rows_delivery(&p, &values, encoded, false, shared);
                match result {
                    Ok((safe, valid, largest)) => {
                        assert_eq!(safe, values);
                        assert!(valid.iter().all(|&v| v == 1));
                        if encoded { assert_eq!(largest, 16 * values.len() + 92); }
                    }
                    Err(error) => assert_eq!(error, Error::Bounds),
                }
            }
            let retained = checked_rows_delivery(&p, &values, true, true, shared);
            if rows >= 3 { assert!(matches!(retained, Err(Error::Bounds))); }
            else if let Err(error) = retained { assert_eq!(error, Error::Bounds); }
            let typed_retained = checked_rows_delivery(&p, &values, false, true, shared);
            if rows >= 3 { assert!(matches!(typed_retained, Err(Error::Bounds))); }
            else if let Err(error) = typed_retained { assert_eq!(error, Error::Bounds); }
        }
    }
    assert_eq!((16 * 512 + 92, 16 * 384 + 92), (8284, 6236));
    // Original full Frames remain charged until real caller drop. This historical
    // frame-payload frontier alone exceeds the unchanged first PPA allowance.
    assert_eq!(28 * 512 * 8, 114_688);
    assert!(114_688 > 94_208);
    // At384, first PPA alone is inconclusive. Retained originals from the
    // first PPA plus the second real copied operand exceed the SECOND planner
    // allowance, even ignoring OT sends and every other retained register.
    assert_eq!((43 + 2) * 384 * 8, 138_240);
    assert!(138_240 > 135_168);
    // Normal-drop H=0 removes those history obstructions. These are thresholds,
    // NOT measured live R_low/R_high, allocation peaks or whole-Executor fit.
    assert_eq!(258_048 - 320 * 512, 94_208);
    assert_eq!(258_048 - 320 * 384, 135_168);
    assert_eq!((94_208 - 16 * 512, 135_168 - 16 * 384), (86_016, 129_024));
}

#[test]
fn checked_rows_scalar_all_valid_executes_global_not_per_row_reduction() {
    for values in [vec![1; 24], { let mut v = vec![1; 24]; v[17] = 0; v }, vec![0; 24]] {
        let p = Program::all_valid(values.len()).unwrap();
        for encoded in [false, true] {
            let (output, validity, _) = checked_rows_delivery(&p, &values, encoded, false, false).unwrap();
            assert_eq!(output, [i64::from(values.iter().all(|&x| x == 1))]);
            assert!(validity.is_empty());
        }
    }
}

#[test]
fn checked_rows_owned_shared_executor_admission_matches_real_full_resource_getters() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    for (rows, width) in [(1, 16), (3, 128), (4, 128), (8, 64)] {
        let program = Program::checked_range_rows(rows, width, full, full, 0).unwrap();
        let resources = program.resources().unwrap();
        for shared in [false, true] {
            let runtimes = crate::puma::multiply::tests::parties(program.work_budget().limits().unwrap());
            let mut inputs = checked_rows_inputs(&runtimes, &vec![0; rows * width]).map(Some);
            let template = Arc::new(program.clone());
            for (party, runtime) in runtimes.into_iter().enumerate() {
                let usage = runtime.payload_usage();
                let input = inputs[party].take().unwrap();
                let credit = input.first.charged_bytes(&usage) + input.second.charged_bytes(&usage);
                let full_resources = if shared { resources.shared_admitted_bytes } else { resources.owned_admitted_bytes };
                let expected = full_resources - FIXED_RESERVE_BYTES + usage.live_bytes() - credit <= usage.limit_bytes();
                let result = if shared { Executor::new_shared(runtime, Arc::clone(&template), input) }
                    else { Executor::new(runtime, program.clone(), input) };
                assert_eq!(result.is_ok(), expected, "rows={rows}, width={width}, shared={shared}, party={party}");
                if let Err(error) = &result { assert_eq!(*error, Error::Bounds); }
                drop(result);
                assert_eq!(usage.live_bytes(), 0);
            }
        }
    }
}

struct CheckedRowsSeeds { party: usize, nonce: u128 }
impl crate::puma::prss::SeedEntropy for CheckedRowsSeeds {
    fn draw(&mut self, purpose: crate::puma::prss::SeedPurpose) -> Result<zeroize::Zeroizing<[u8; 16]>, Error> {
        use crate::puma::prss::SeedPurpose;
        let value = match purpose {
            SeedPurpose::Public => self.nonce * 4 + self.party as u128 + 1,
            SeedPurpose::Private => self.party as u128 + 101,
            SeedPurpose::Pairwise => (self.party as u128 + 1) * 11,
        };
        Ok(zeroize::Zeroizing::new(value.to_le_bytes()))
    }
}

fn checked_rows_seeded_parties(work: WorkBudget, nonce: u128) -> [PartyRuntime; 3] {
    let context = crate::puma::multiply::bound_context(Context::new([31; 32], [43; 32], 7).unwrap());
    let mut setups = [Party::P0, Party::P1, Party::P2].map(|party| {
        crate::puma::multiply::SetupParty(crate::puma::prss::SetupParty::with_entropy(context, party,
            work.limits().unwrap(), &mut CheckedRowsSeeds { party: party as usize, nonce }).unwrap())
    });
    let public = setups.each_mut().map(|p| p.contribution().unwrap());
    for (i, owner) in setups.iter_mut().enumerate() {
        for (j, contribution) in public.iter().enumerate() {
            if i != j { owner.receive_public(*contribution).unwrap(); }
        }
    }
    let mut seeds = setups.each_mut().map(|p| Some(p.share_seed().unwrap()));
    for (i, owner) in setups.iter_mut().enumerate() {
        owner.receive_seed(seeds[(i + 1) % 3].take().unwrap()).unwrap();
    }
    setups.each_mut().map(|p| p.finish().unwrap())
}

#[test]
fn checked_rows_all_public_ot_pivots_and_roles_on_typed_and_encoded_delivery() {
    let full = Interval::new(i64::MIN, i64::MAX).unwrap();
    let accepted = Interval::new(-1, 1).unwrap();
    let values = [i64::MIN, -1, 0, 1, i64::MAX, 2];
    let p = Program::checked_range_rows(2, 3, full, accepted, 0).unwrap();
    for encoded in [false, true] {
        for shared in [false, true] {
            let mut helpers = [false; 3];
            // Ordinary deterministic TEST entropy, not a production pivot setter.
            // Finite fixtures must actually observe all pivots; no role claim is
            // accepted merely because WorkBudget prices their upper envelope.
            for nonce in 0..32 {
                let runtimes = checked_rows_seeded_parties(p.work_budget(), nonce);
                let (safe, valid, _) = checked_rows_delivery_with_runtimes(&p, &values,
                    encoded, false, shared, runtimes, &mut helpers, None).unwrap();
                assert_eq!(safe, [0, -1, 0, 1, 0, 0]);
                assert_eq!(valid, [0, 1, 1, 1, 0, 0]);
                if helpers.iter().all(|&seen| seen) { break; }
            }
            assert_eq!(helpers, [true; 3]);
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Rms {
    Plain,
    Certified,
    InputNorm(u128),
}
impl Rms {
    fn work(
        self,
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
        space: &mut RmsWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        match self {
            Self::Plain => Program::rms_norm_work(rows, width, input, gamma, space),
            Self::Certified => Program::rms_norm_certified_work(rows, width, input, gamma, space),
            Self::InputNorm(norm) => {
                Program::rms_norm_with_input_norm_work(rows, width, input, gamma, norm, space)
            }
        }
    }
    fn native(
        self,
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
    ) -> Result<(Program, u128), Error> {
        match self {
            Self::Plain => Program::rms_norm(rows, width, input, gamma).map(|p| (p, 0)),
            Self::Certified => Program::rms_norm_certified(rows, width, input, gamma),
            Self::InputNorm(norm) => {
                Program::rms_norm_with_input_norm(rows, width, input, gamma, norm)
            }
        }
    }
}

// Literal original OUTER composition, not rms_recipe or a work query. The
// normalization proof remains the native owner's shared primitive (not an
// independently reimplemented numeric/certificate recipe).
fn original_outer(
    kind: Rms,
    rows: usize,
    width: usize,
    input: Interval,
    gamma: &[i64],
) -> Result<(Program, u128), Error> {
    let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
    if rows == 0 || width == 0 || gamma.len() != width {
        return Err(Error::Shape);
    }
    let mut b = Builder::row(lanes, input)?;
    b.0.numeric_profile = RMS_NORM_PROFILE;
    let partitioned = !matches!(kind, Rms::Plain);
    let norm = match kind {
        Rms::InputNorm(v) => Some(v),
        _ => None,
    };
    let (normalized, radius) = numeric::rms_normalization(&mut b, width, input, partitioned, norm)?;
    let scales: [Affine; MAX_ROW_LANES] = std::array::from_fn(|i| Affine {
        left: gamma[i % width],
        right: 0,
        constant: 0,
    });
    let scaled = b.affine_lanes(normalized, &scales[..lanes])?;
    let output = b.rescale(scaled, 18)?;
    fn root(v: u128) -> u128 {
        let r = v.isqrt();
        r + u128::from(r * r != v)
    }
    let energy = if partitioned {
        let g = gamma
            .iter()
            .map(|v| v.unsigned_abs())
            .max()
            .ok_or(Error::Shape)?;
        let r = radius
            .checked_mul(u128::from(g))
            .ok_or(Error::Bounds)?
            .div_ceil(1 << 18)
            .checked_add(root(width as u128))
            .ok_or(Error::Bounds)?;
        r.checked_mul(r).ok_or(Error::Bounds)?
    } else {
        0
    };
    Ok((b.finish(output)?, energy))
}

#[test]
fn rms_work_native_metadata_outer_composition_gamma_columns_and_certificates_match() {
    let mut space = RmsWorkWorkspace::empty();
    let mut successes = [0; 3];
    for rows in [1, 2] {
        for width in [1, 3, 7, 128, MAX_ROW_LANES] {
            for input in [
                Interval::new(0, 0).unwrap(),
                Interval::new(-32, 32).unwrap(),
                Interval::new(-(1 << 18), 1 << 18).unwrap(),
            ] {
                let magnitude = u128::from(
                    input
                        .lower()
                        .unsigned_abs()
                        .max(input.upper().unsigned_abs()),
                );
                let obligation = magnitude * magnitude * width as u128;
                let gamma: Vec<_> = (0..width)
                    .map(|i| match i % 3 {
                        0 => 1 << 18,
                        1 => -(1 << 17),
                        _ => 2 << 18,
                    })
                    .collect();
                for (index, kind) in [Rms::Plain, Rms::Certified, Rms::InputNorm(obligation)]
                    .into_iter()
                    .enumerate()
                {
                    space.reset();
                    let before = attempts();
                    let q = kind.work(rows, width, input, &gamma, &mut space);
                    assert_eq!(attempts(), before);
                    let a = kind.native(rows, width, input, &gamma);
                    let e = original_outer(kind, rows, width, input, &gamma);
                    match (q, a, e) {
                        (Ok(q), Ok((a, ae)), Ok((e, ee))) => {
                            successes[index] += 1;
                            space.assert_native_metadata(&a);
                            assert_eq!(q, a.work_budget());
                            assert_eq!(q, e.work_budget());
                            assert_eq!(ae, ee);
                            assert_eq!(a.nodes, e.nodes);
                            assert_eq!(a.digest(), e.digest());
                            assert_eq!(a.numeric_profile(), e.numeric_profile());
                            assert_eq!(a.last_use, e.last_use);
                            assert_eq!(a.slots, e.slots);
                            assert_eq!(a.register_count, e.register_count);
                            assert_eq!(a.public_affines, e.public_affines);
                            assert_eq!(a.public_affine_count, 1);
                            let table = &a.public_affines[0];
                            assert_eq!(table.len(), rows * width);
                            assert_eq!(crate::exact_buffer::layout_bytes::<Affine>(table.len()).unwrap(),
                                rows * width * std::mem::size_of::<Affine>());
                            for (i, c) in table.iter().enumerate() {
                                assert_eq!((c.left, c.right, c.constant), (gamma[i % width], 0, 0));
                            }
                        }
                        (Err(q), Err(a), Err(e)) => {
                            assert_eq!(q, a);
                            assert_eq!(a, e);
                            assert!(space.is_poisoned());
                        }
                        _ => panic!(
                            "RMS work/native/outer mismatch {kind:?}, {rows}x{width}, {input:?}"
                        ),
                    }
                }
            }
        }
    }
    assert!(successes.into_iter().all(|n| n > 0));
}

#[test]
fn rms_work_parameter_norm_overflow_and_competing_error_precedence_poison_then_reset() {
    let mut space = RmsWorkWorkspace::empty();
    let cases = [
        (
            usize::MAX,
            2,
            Interval::new(0, 0).unwrap(),
            &[][..],
            Rms::Certified,
            Error::Bounds,
        ),
        (
            MAX_ROW_LANES + 1,
            1,
            Interval::new(i64::MIN, i64::MAX).unwrap(),
            &[][..],
            Rms::InputNorm(u128::MAX),
            Error::Shape,
        ),
        (
            0,
            0,
            Interval::new(0, 0).unwrap(),
            &[][..],
            Rms::Plain,
            Error::Shape,
        ),
        (
            1,
            1,
            Interval::new(i64::MIN, i64::MAX).unwrap(),
            &[1_i64][..],
            Rms::Certified,
            Error::Bounds,
        ),
        (
            1,
            1,
            Interval::new(1 << 18, 1 << 18).unwrap(),
            &[1_i64][..],
            Rms::InputNorm(0),
            Error::Bounds,
        ),
        (
            1,
            1,
            Interval::new(0, 0).unwrap(),
            &[1_i64][..],
            Rms::InputNorm(u128::MAX),
            Error::Bounds,
        ),
        (
            1,
            1,
            Interval::new(1 << 18, 1 << 18).unwrap(),
            &[i64::MAX][..],
            Rms::Plain,
            Error::Bounds,
        ),
    ];
    for (rows, width, input, gamma, kind, error) in cases {
        space.reset();
        let before = attempts();
        assert!(matches!(kind.work(rows,width,input,gamma,&mut space),Err(e) if e==error));
        assert_eq!(attempts(), before);
        assert!(matches!(kind.native(rows,width,input,gamma),Err(e) if e==error));
        assert!(space.is_poisoned());
        let before = attempts();
        for _ in 0..3 {
            assert!(matches!(
                Program::all_valid_work(1, &mut space),
                Err(Error::Phase)
            ));
            assert!(matches!(
                Program::rms_norm_work(usize::MAX, 2, input, &[], &mut space),
                Err(Error::Phase)
            ));
        }
        assert_eq!(attempts(), before);
        space.reset();
        assert!(!space.is_poisoned());
        let mut fresh = RmsWorkWorkspace::empty();
        assert_eq!(
            Program::rms_norm_work(1, 1, Interval::new(0, 0).unwrap(), &[1], &mut space),
            Program::rms_norm_work(1, 1, Interval::new(0, 0).unwrap(), &[1], &mut fresh)
        );
    }
}

#[test]
fn checked_range_and_all_valid_are_distinct_native_factories_with_exact_work_and_caps() {
    let mut space = RmsWorkWorkspace::empty();
    for lanes in [1, 3, 127, 128] {
        for accepted in [
            Interval::new(-32, 32).unwrap(),
            Interval::new(i64::MIN, i64::MAX).unwrap(),
        ] {
            let incoming = Interval::new(i64::MIN, i64::MAX).unwrap();
            let before = attempts();
            let checked =
                Program::checked_range_work(lanes, incoming, accepted, 0, &mut space).unwrap();
            assert_eq!(attempts(), before);
            let program = Program::checked_range(lanes, incoming, accepted, 0).unwrap();
            space.assert_native_metadata(&program);
            assert_eq!(checked, program.work_budget());
            assert_eq!(program.output_bounds(), accepted);
            assert!(program.checked_validity.is_some());
            assert_eq!(program.numeric_profile(), CHECKED_RANGE_PROFILE);
            let before = attempts();
            let reduce = Program::all_valid_work(lanes, &mut space).unwrap();
            assert_eq!(attempts(), before);
            let p = Program::all_valid(lanes).unwrap();
            space.assert_native_metadata(&p);
            assert_eq!(reduce, p.work_budget());
            assert!(p.checked_validity.is_none());
            assert_ne!(checked, reduce);
            let total = checked.checked_add(reduce).unwrap();
            assert!(total
                .admit(
                    0,
                    crate::puma::prss::Cursors::default(),
                    total.operations() - 1,
                    total.cursors().aes_work
                )
                .is_err());
        }
    }
    // Reduction uses a complete row, not Builder::new's tile cap.
    let q = Program::all_valid_work(MAX_ROW_LANES, &mut space).unwrap();
    assert_eq!(q, Program::all_valid(MAX_ROW_LANES).unwrap().work_budget());
    let input = Interval::new(-32, 32).unwrap();
    let before = attempts();
    assert!(matches!(
        Program::checked_range_work(129, input, input, 0, &mut space),
        Err(Error::Bounds)
    ));
    assert_eq!(attempts(), before);
    assert!(matches!(
        Program::all_valid_work(1, &mut space),
        Err(Error::Phase)
    ));
    space.reset();
    assert!(matches!(
        Program::checked_range_work(1, input, input, 33, &mut space),
        Err(Error::Bounds)
    ));
    space.reset();
    assert!(matches!(
        Program::all_valid_work(0, &mut space),
        Err(Error::Bounds)
    ));
    space.reset();
    assert!(matches!(
        Program::all_valid_work(MAX_ROW_LANES + 1, &mut space),
        Err(Error::Bounds)
    ));
}

fn full_tail(space: &mut RmsWorkWorkspace) -> Result<WorkBudget, Error> {
    let input = Interval::new(-32, 32)?;
    Program::rms_norm_certified_work(2, 3, input, &[1 << 18; 3], space)?
        .checked_mul(2)?
        .checked_add(Program::rms_norm_certified_work(
            1,
            3,
            input,
            &[1 << 18; 3],
            space,
        )?)
}
fn admitted(space: &mut RmsWorkWorkspace, ops: u64, aes: u64) -> Result<(), Error> {
    let q = full_tail(space)?;
    q.admit(0, crate::puma::prss::Cursors::default(), ops, aes)?;
    Program::rms_norm_certified(2, 3, Interval::new(-32, 32)?, &[1 << 18; 3])?;
    Program::rms_norm_certified(1, 3, Interval::new(-32, 32)?, &[1 << 18; 3])?;
    Ok(())
}

#[test]
fn rms_full_tail_pre_entry_exact_short_caps_checked_aggregation_and_reset_do_not_refund() {
    let mut space = RmsWorkWorkspace::empty();
    let before = attempts();
    let q = full_tail(&mut space).unwrap();
    assert_eq!(attempts(), before);
    for (ops, aes) in [
        (0, q.cursors().aes_work),
        (q.operations() - 1, q.cursors().aes_work),
        (q.operations(), 0),
        (q.operations(), q.cursors().aes_work - 1),
    ] {
        assert!(matches!(admitted(&mut space, ops, aes), Err(Error::Bounds)));
        assert_eq!(attempts(), before);
    }
    admitted(&mut space, q.operations(), q.cursors().aes_work).unwrap();
    let after = attempts();
    assert_eq!(after[0], before[0] + 2);
    assert!(matches!(q.checked_mul(u64::MAX), Err(Error::Bounds)));
    assert!(matches!(
        WorkBudget::local()
            .checked_mul(u64::MAX)
            .unwrap()
            .checked_add(q),
        Err(Error::Bounds)
    ));
    for c in [
        crate::puma::prss::Cursors {
            public_blocks: u64::MAX,
            ..Default::default()
        },
        crate::puma::prss::Cursors {
            pair_blocks: u64::MAX,
            ..Default::default()
        },
        crate::puma::prss::Cursors {
            private_blocks: u64::MAX,
            ..Default::default()
        },
        crate::puma::prss::Cursors {
            aes_work: u64::MAX,
            ..Default::default()
        },
    ] {
        assert!(q.admit(0, c, u64::MAX, u64::MAX).is_err());
    }
    assert_eq!(attempts(), after);
    let full = Program::rms_norm_certified(2, 3, Interval::new(-32, 32).unwrap(), &[1 << 18; 3])
        .unwrap()
        .0
        .work_budget();
    let tail = Program::rms_norm_certified(1, 3, Interval::new(-32, 32).unwrap(), &[1 << 18; 3])
        .unwrap()
        .0
        .work_budget();
    assert_eq!(q, full.checked_mul(2).unwrap().checked_add(tail).unwrap());
    let merged = Program::rms_norm_certified_work(
        5,
        3,
        Interval::new(-32, 32).unwrap(),
        &[1 << 18; 3],
        &mut space,
    )
    .unwrap();
    assert_ne!(q.cursors(), merged.cursors());
    let prior = q;
    space.reset();
    assert_eq!(
        q.checked_add(full_tail(&mut space).unwrap()).unwrap(),
        prior.checked_mul(2).unwrap()
    );
}

#[test]
fn rms_workspace_quote_before_constructor_is_actual_inline_capacity_not_budget_authority() {
    fn construct(limit: usize, calls: &mut usize) -> Result<RmsWorkWorkspace, Error> {
        if RmsWorkWorkspace::workspace_bytes() > limit {
            return Err(Error::Bounds);
        }
        *calls += 1;
        Ok(RmsWorkWorkspace::empty())
    }
    let quote = RmsWorkWorkspace::workspace_bytes();
    assert_eq!(quote, std::mem::size_of::<RmsWorkWorkspace>());
    assert!(
        quote
            >= SoftmaxWorkWorkspace::workspace_bytes()
                + std::mem::size_of::<usize>()
                + std::mem::size_of::<bool>()
    );
    let mut calls = 0;
    assert!(construct(quote - 1, &mut calls).is_err());
    assert_eq!(calls, 0);
    let mut space = construct(quote, &mut calls).unwrap();
    assert_eq!(calls, 1);
    assert!(
        Program::rms_norm_certified_work(1, 1, Interval::new(0, 0).unwrap(), &[1], &mut space)
            .is_ok()
    );
    // No complete constructor/controller/proof-stack/RSS/input-source quote.
}
