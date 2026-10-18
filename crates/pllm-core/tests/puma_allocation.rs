use pllm_core::puma::prss::{PrssLimits, SetupParty, SourceParty};
use pllm_core::puma::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
    static FAIL_NEXT: Cell<bool> = const { Cell::new(false) };
    static FAIL_AT: Cell<Option<usize>> = const { Cell::new(None) };
    static AUDIT: Cell<Audit> = const { Cell::new(Audit::empty()) };
}

#[derive(Clone, Copy)]
struct Slot {
    address: usize,
    bytes: usize,
}
#[derive(Clone, Copy)]
struct Audit {
    active: bool,
    collect: bool,
    exact: bool,
    usage: *const PayloadUsage,
    slots: [Slot; 256],
    allocations: usize,
    frees: usize,
    first_live: usize,
    bad_erasure: bool,
    bad_charge: bool,
    overflow: bool,
    reallocations: usize,
}
impl Audit {
    const fn empty() -> Self {
        Self {
            active: false,
            collect: false,
            exact: false,
            usage: std::ptr::null(),
            slots: [Slot {
                address: 0,
                bytes: 0,
            }; 256],
            allocations: 0,
            frees: 0,
            first_live: 0,
            bad_erasure: false,
            bad_charge: false,
            overflow: false,
            reallocations: 0,
        }
    }
    fn remaining(&self) -> usize {
        self.slots.iter().map(|s| s.bytes).sum()
    }
    fn add(&mut self, address: usize, bytes: usize) {
        if let Some(slot) = self.slots.iter_mut().find(|s| s.address == 0) {
            *slot = Slot { address, bytes };
            self.allocations += 1;
        } else {
            self.overflow = true;
        }
    }
}
struct Watch;
impl Watch {
    fn start(usage: &PayloadUsage, collect: bool, exact: bool) -> Self {
        AUDIT.set(Audit {
            active: true,
            collect,
            exact,
            usage,
            ..Audit::empty()
        });
        Self
    }
    fn finish(self) -> Audit {
        let result = AUDIT.get();
        drop(self);
        result
    }
}
impl Drop for Watch {
    fn drop(&mut self) {
        AUDIT.set(Audit::empty());
    }
}
fn fail_allocation() -> bool {
    if FAIL_NEXT.try_with(|v| v.replace(false)).unwrap_or(false) {
        return true;
    }
    FAIL_AT
        .try_with(|v| match v.get() {
            Some(0) => {
                v.set(None);
                true
            }
            Some(n) => {
                v.set(Some(n - 1));
                false
            }
            None => false,
        })
        .unwrap_or(false)
}
fn record_allocation(ptr: *mut u8, bytes: usize) {
    let _ = AUDIT.try_with(|cell| {
        let mut a = cell.get();
        if a.active && a.collect && !ptr.is_null() {
            // Watch guards keep this stack-resident read-only handle alive.
            let live = unsafe { &*a.usage }.live_bytes();
            if a.allocations == 0 {
                a.first_live = live;
            }
            a.add(ptr as usize, bytes);
            a.bad_charge |= live < a.remaining();
            cell.set(a);
        }
    });
}
fn assert_audit(a: Audit) {
    assert!(!a.bad_erasure, "payload not zero at deallocation");
    assert!(
        !a.bad_charge,
        "payload uncharged before allocation/deallocation"
    );
    assert!(!a.overflow);
    assert_eq!(a.reallocations, 0);
    assert_eq!(a.frees, a.allocations);
    assert_eq!(a.remaining(), 0);
}

struct Counting;
// Test-only instrumentation. Thread-local flags avoid interference from the
// harness's other threads. Successful calls delegate to the system allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if fail_allocation() {
            return std::ptr::null_mut();
        }
        if TRACK.try_with(Cell::get).unwrap_or(false) {
            let _ = COUNT.try_with(|v| v.set(v.get() + layout.size()));
        }
        let ptr = unsafe { System.alloc(layout) };
        record_allocation(ptr, layout.size());
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let before_free = AUDIT
            .try_with(|cell| {
                let mut a = cell.get();
                if a.active {
                    if let Some(i) = a.slots.iter().position(|s| s.address == ptr as usize) {
                        let bytes = a.slots[i].bytes;
                        a.bad_erasure |= layout.size() != bytes;
                        // Only fully initialized u64 payload allocations are watched.
                        a.bad_erasure |= unsafe { std::slice::from_raw_parts(ptr, bytes) }
                            .iter()
                            .any(|&v| v != 0);
                        let live = unsafe { &*a.usage }.live_bytes();
                        a.bad_charge |= if a.exact {
                            live != a.remaining()
                        } else {
                            live < a.remaining()
                        };
                        a.slots[i] = Slot {
                            address: 0,
                            bytes: 0,
                        };
                        a.frees += 1;
                        cell.set(a);
                        return Some((a.usage, live));
                    }
                }
                None
            })
            .ok()
            .flatten();
        unsafe { System.dealloc(ptr, layout) }
        // The lease must still exist even after System.dealloc returns inside
        // this callback. Refund is allowed only after the callback itself ends.
        if let Some((usage, live)) = before_free {
            let _ = AUDIT.try_with(|cell| {
                let mut a = cell.get();
                a.bad_charge |= unsafe { &*usage }.live_bytes() != live;
                cell.set(a);
            });
        }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, n: usize) -> *mut u8 {
        if fail_allocation() {
            return std::ptr::null_mut();
        }
        if TRACK.try_with(Cell::get).unwrap_or(false) {
            let _ = COUNT.try_with(|v| v.set(v.get() + n));
        }
        let _ = AUDIT.try_with(|c| {
            let mut a = c.get();
            if a.active {
                a.reallocations += 1;
                c.set(a);
            }
        });
        unsafe { System.realloc(ptr, layout, n) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn setup() -> (Context, Operation) {
    let context = Context::new([5; 32], [6; 32], 7).unwrap();
    (
        context,
        Operation::new(context, 0, Party::P0, 2, 18, Limits::default()).unwrap(),
    )
}

fn measured<T>(f: impl FnOnce() -> T) -> (T, usize) {
    COUNT.set(0);
    TRACK.set(true);
    let result = f();
    TRACK.set(false);
    (result, COUNT.get())
}

#[test]
fn shape_limit_and_bad_binding_reject_before_payload_allocation() {
    let (context, op) = setup();
    let limits = Limits::default();
    let (_, bytes) = measured(|| {
        assert!(Operation::new(context, 0, Party::P0, usize::MAX, 18, limits).is_err());
        assert!(LocalMaterial::helper(op, &[0; 2], &[0; 1], &[0; 2], &[0; 2], limits).is_err());
        assert!(ReplicatedShare::from_local(op, Party::P0, &[0; 2], &[0; 1], limits).is_err());
        let header = Header {
            binding: [9; 32],
            from: Party::P0,
            to: Party::P1,
            phase: MessagePhase::MaskedInput,
        };
        assert!(Message::from_received(header, &[0; 2], op, limits).is_err());
        let header = Header {
            binding: op.digest(),
            ..header
        };
        assert!(Message::from_received(header, &[0; 3], op, limits).is_err());
    });
    assert_eq!(bytes, 0);
}

#[test]
fn allocator_failure_after_burn_is_fallible_and_terminal() {
    let (context, op) = setup();
    let limits = Limits::default();
    let mut runtime = PartyRuntime::new(context, Party::P0, limits);
    let input = ReplicatedShare::from_local(op, Party::P0, &[0; 2], &[0; 2], limits).unwrap();
    let material = LocalMaterial::evaluator(op, Party::P0, &[1; 2], &[2; 2], limits).unwrap();
    FAIL_NEXT.set(true);
    let result = runtime.begin(op, input, material);
    FAIL_NEXT.set(false);
    assert!(matches!(result, Err(Error::Allocation)));
    assert_eq!(runtime.status(), Status::Retired);
    assert_eq!(runtime.next_sequence(), 1);
}

#[test]
fn narrower_runtime_admission_burns_without_allocating() {
    let (context, op) = setup();
    let limits = Limits::default();
    let narrow = Limits::new(1, 1, 8192).unwrap();
    let mut runtime = PartyRuntime::new(context, Party::P0, narrow);
    let input = ReplicatedShare::from_local(op, Party::P0, &[0; 2], &[0; 2], limits).unwrap();
    let material = LocalMaterial::evaluator(op, Party::P0, &[1; 2], &[2; 2], limits).unwrap();
    let (result, bytes) = measured(|| runtime.begin(op, input, material));
    assert!(matches!(result, Err(Error::Bounds)));
    assert_eq!(bytes, 0);
    assert_eq!(runtime.status(), Status::Retired);
    assert_eq!(runtime.next_sequence(), 1);
}

#[test]
fn helper_payload_allocations_fit_declared_conservative_reserve() {
    let limits = Limits::default();
    let (context, _) = setup();
    let op = Operation::new(context, 0, Party::P0, MAX_LANES, 18, limits).unwrap();
    let words = [0_u64; MAX_LANES];
    // The one fixed-size ledger Arc is an explicit construction exception.
    let mut runtime = PartyRuntime::new(context, Party::P2, limits);
    let usage = runtime.payload_usage();
    let (_, bytes) = measured(|| {
        let input = ReplicatedShare::from_local(op, Party::P2, &words, &words, limits).unwrap();
        let material = LocalMaterial::helper(op, &words, &words, &words, &words, limits).unwrap();
        runtime
            .begin(op, input, material)
            .unwrap_or_else(|e| panic!("{e}"));
        runtime.take_output().unwrap_or_else(|e| panic!("{e}"));
    });
    assert_eq!(bytes, 10 * MAX_LANES * 8);
    assert_eq!(usage.peak_bytes(), 10 * MAX_LANES * 8);
    assert_eq!(usage.live_bytes(), 0);
    assert!(bytes <= limits.admitted_bytes(MAX_LANES).unwrap());
    assert_eq!(limits.role_working_bytes(op, Party::P2), Ok(4096 + bytes));
    assert_eq!(
        limits.role_working_bytes(op, Party::P0),
        Ok(4096 + 8 * MAX_LANES * 8)
    );
}

// Test-only aggregate transport. Setup itself issues only one role's edge seed.
fn source_owners(limits: PrssLimits) -> [SourceParty; 3] {
    let (context, _) = setup();
    let mut setups =
        [Party::P0, Party::P1, Party::P2].map(|p| SetupParty::new(context, p, limits).unwrap());
    let contributions = setups.each_mut().map(|p| p.contribution().unwrap());
    for (i, owner) in setups.iter_mut().enumerate() {
        for (j, contribution) in contributions.iter().enumerate() {
            if i != j {
                owner.receive_public(*contribution).unwrap();
            }
        }
    }
    let mut seeds = setups.each_mut().map(|p| Some(p.share_seed().unwrap()));
    for (i, owner) in setups.iter_mut().enumerate() {
        owner
            .receive_seed(seeds[(i + 1) % 3].take().unwrap())
            .unwrap();
    }
    setups.each_mut().map(|p| p.finish().unwrap())
}

#[test]
fn source_setup_uses_only_fixed_ledger_metadata_and_invalid_plan_allocates_nothing() {
    let (mut owners, bytes) = measured(|| source_owners(PrssLimits::default()));
    assert!(bytes > 0 && bytes <= 3 * 4096);
    let (result, bytes) = measured(|| owners[0].plan(usize::MAX, 18));
    assert_eq!(result, Err(Error::Bounds));
    assert_eq!(bytes, 0);
    assert_eq!(owners[0].cursors().aes_work, 0);
    assert_eq!(owners[0].status(), Status::Retired);
}

#[test]
fn source_allocation_failure_burns_reserved_ranges_and_retires_owner() {
    let mut owners = source_owners(PrssLimits::default());
    let op = owners[0].plan(3, 18).unwrap();
    let input =
        ReplicatedShare::from_local(op, Party::P0, &[0; 3], &[0; 3], Limits::default()).unwrap();
    FAIL_NEXT.set(true);
    let result = owners[0].begin(input);
    FAIL_NEXT.set(false);
    assert!(matches!(result, Err(Error::Allocation)));
    assert_eq!(owners[0].cursors().pair_blocks, 4);
    assert_eq!(owners[0].status(), Status::Retired);
    assert!(matches!(owners[0].plan(3, 18), Err(Error::Retired)));
}

#[test]
fn source_work_exhaustion_precedes_any_material_allocation() {
    let mut owners = source_owners(PrssLimits::new(Limits::default(), 1).unwrap());
    let op = owners[0].plan(3, 18).unwrap();
    let input =
        ReplicatedShare::from_local(op, Party::P0, &[0; 3], &[0; 3], Limits::default()).unwrap();
    let (result, bytes) = measured(|| owners[0].begin(input));
    assert!(matches!(result, Err(Error::Bounds)));
    assert_eq!(bytes, 0);
    assert_eq!(owners[0].cursors().pair_blocks, 0);
    assert_eq!(owners[0].status(), Status::Retired);
}

fn helper_inputs(sequence: u64) -> (Operation, ReplicatedShare, LocalMaterial) {
    let (context, _) = setup();
    let limits = Limits::default();
    let op = Operation::new(context, sequence, Party::P0, MAX_LANES, 18, limits).unwrap();
    // Public fixture only: not a production deterministic dealer.
    let values = [sequence + 1; MAX_LANES];
    let input = ReplicatedShare::from_local(op, Party::P2, &values, &values, limits).unwrap();
    let material = LocalMaterial::helper(op, &values, &values, &values, &values, limits).unwrap();
    (op, input, material)
}

#[test]
fn retained_original_frames_and_outputs_exhaust_without_resetting_on_extraction() {
    for retain_outputs in [false, true] {
        let (context, _) = setup();
        let mut owner = PartyRuntime::new(context, Party::P2, Limits::default());
        let usage = owner.payload_usage();
        let mut frames = Vec::with_capacity(16);
        let mut outputs = Vec::with_capacity(16);
        let mut completed = 0;
        for sequence in 0..16 {
            let (op, input, material) = helper_inputs(sequence);
            match owner.begin(op, input, material) {
                Ok(messages) => {
                    let output = owner.take_output().unwrap();
                    if retain_outputs {
                        outputs.push(output);
                        drop(messages);
                    } else {
                        frames.push(messages);
                        drop(output);
                    }
                    completed += 1;
                    assert_eq!(
                        usage.live_bytes(),
                        completed * MAX_LANES * 8 * if retain_outputs { 2 } else { 4 }
                    );
                }
                Err(error) => {
                    assert_eq!(error, Error::Bounds);
                    assert_eq!(owner.status(), Status::Retired);
                    assert_eq!(owner.next_sequence(), sequence + 1);
                    break;
                }
            }
        }
        assert_eq!(completed, if retain_outputs { 11 } else { 6 });
        assert!(usage.peak_bytes() <= usage.limit_bytes());
        assert_eq!(usage.limit_bytes(), MAX_ADMITTED_BYTES - 4096);
        let retained = usage.live_bytes();
        owner.cancel();
        assert_eq!(usage.live_bytes(), retained);
        drop(frames);
        drop(outputs);
        assert_eq!(usage.live_bytes(), 0);
        // Refunded storage does not revive a retired protocol owner.
        let (op, input, material) = helper_inputs(completed as u64 + 1);
        assert!(matches!(
            owner.begin(op, input, material),
            Err(Error::Retired)
        ));
    }
}

#[test]
fn releasing_retained_frames_refunds_capacity_for_later_ordinals() {
    let (context, _) = setup();
    let mut owner = PartyRuntime::new(context, Party::P2, Limits::default());
    let usage = owner.payload_usage();
    let mut retained = Vec::with_capacity(6);
    for sequence in 0..12 {
        if retained.len() == 6 {
            drop(retained.pop());
            assert_eq!(usage.live_bytes(), 5 * 4 * MAX_LANES * 8);
        }
        let (op, input, material) = helper_inputs(sequence);
        retained.push(owner.begin(op, input, material).unwrap());
        drop(owner.take_output().unwrap());
    }
    assert_eq!(owner.next_sequence(), 12);
    assert_eq!(usage.live_bytes(), 6 * 4 * MAX_LANES * 8);
    drop(owner);
    assert_eq!(usage.live_bytes(), 6 * 4 * MAX_LANES * 8);
    drop(retained);
    assert_eq!(usage.live_bytes(), 0);
}

#[test]
fn source_exported_payloads_are_wiped_and_still_charged_at_deallocation() {
    let mut owners = source_owners(PrssLimits::default());
    let ops = owners.each_mut().map(|p| p.plan(3, 18).unwrap());
    assert_eq!(ops, [ops[0]; 3]);
    let op = ops[0];
    let h = op.helper() as usize;
    let usage = owners[h].payload_usage();
    let input = ReplicatedShare::from_local(op, op.helper(), &[21; 3], &[22; 3], Limits::default())
        .unwrap();
    let frames = owners[h].begin(input).unwrap();
    let output = owners[h].take_output().unwrap();
    assert_eq!(usage.peak_bytes(), 10 * 3 * 8); // original input pair IS priced
    assert_eq!(usage.live_bytes(), 6 * 3 * 8);
    let watch = Watch::start(&usage, false, true);
    let mut audit = AUDIT.get();
    for frame in frames.iter().flatten() {
        audit.add(frame.words().as_ptr() as usize, frame.words().len() * 8);
    }
    for words in [output.first(), output.second()] {
        audit.add(words.as_ptr() as usize, words.len() * 8);
    }
    AUDIT.set(audit);
    drop(owners);
    assert_eq!(usage.live_bytes(), 6 * 3 * 8);
    drop(frames);
    assert_eq!(usage.live_bytes(), 2 * 3 * 8);
    drop(output);
    assert_eq!(usage.live_bytes(), 0);
    assert_audit(watch.finish());
}

#[test]
fn every_source_payload_allocation_index_rolls_back_after_earlier_expansions() {
    // Both evaluator roles and H; include the successful no-failure endpoint.
    for role in 0..3 {
        let allocations = if role == 2 { 6 } else { 4 };
        for fail_at in 0..=allocations {
            let mut owners = source_owners(PrssLimits::default());
            let ops = owners.each_mut().map(|p| p.plan(3, 18).unwrap());
            let op = ops[0];
            let party = [op.evaluator_a(), op.evaluator_b(), op.helper()][role];
            let owner = &mut owners[party as usize];
            let usage = owner.payload_usage();
            let input =
                ReplicatedShare::from_local(op, party, &[21; 3], &[22; 3], Limits::default())
                    .unwrap();
            let watch = Watch::start(&usage, true, false);
            FAIL_AT.set(Some(fail_at));
            let result = owner.begin(input);
            FAIL_AT.set(None);
            if fail_at < allocations {
                assert!(matches!(result, Err(Error::Allocation)));
                assert_eq!(owner.status(), Status::Retired);
            } else {
                assert!(result.is_ok());
            }
            drop(result);
            owner.cancel();
            let audit = watch.finish();
            assert_eq!(audit.allocations, fail_at.min(allocations));
            if fail_at > 0 {
                assert_eq!(audit.first_live, 3 * 3 * 8);
            }
            assert_audit(audit);
            assert_eq!(usage.live_bytes(), 0);
            assert!(usage.peak_bytes() <= usage.limit_bytes());
            assert_eq!(owner.cursors().pair_blocks, 4);
            assert_eq!(
                owner.cursors().private_blocks,
                if role == 2 { 3 } else { 0 }
            );
            assert!(matches!(owner.plan(3, 18), Err(Error::Retired)));
        }
    }
}

#[test]
fn open_allocation_sweep_prices_received_frames_and_preserves_issued_leases() {
    for evaluator in 0..2 {
        for fail_at in 0..=2 {
            let mut owners = source_owners(PrssLimits::default());
            let ops = owners.each_mut().map(|p| p.plan(3, 18).unwrap());
            let op = ops[0];
            let mut messages = Vec::with_capacity(6);
            for owner in &mut owners {
                let input = ReplicatedShare::from_local(
                    op,
                    owner.party(),
                    &[0; 3],
                    &[0; 3],
                    Limits::default(),
                )
                .unwrap();
                messages.extend(owner.begin(input).unwrap().into_iter().flatten());
            }
            let party = [op.evaluator_a(), op.evaluator_b()][evaluator];
            let take = |messages: &mut Vec<Message>, phase| {
                let at = messages
                    .iter()
                    .position(|m| m.header().to == party && m.header().phase == phase)
                    .unwrap();
                messages.swap_remove(at)
            };
            let peer = take(&mut messages, MessagePhase::MaskedInput);
            let corr = take(&mut messages, MessagePhase::Correlation);
            let owner = &mut owners[party as usize];
            let cursors = owner.cursors();
            let usage = owner.payload_usage();
            assert_eq!(usage.live_bytes(), 3 * 3 * 8);
            let watch = Watch::start(&usage, true, false);
            FAIL_AT.set(Some(fail_at));
            let result = owner.open(peer, corr);
            FAIL_AT.set(None);
            if fail_at < 2 {
                assert!(matches!(result, Err(Error::Allocation)));
                assert_eq!(owner.status(), Status::Retired);
            } else {
                assert!(result.is_ok());
            }
            drop(result);
            owner.cancel();
            let audit = watch.finish();
            assert_eq!(audit.allocations, fail_at.min(2));
            // Own state + retained original outgoing frame + incoming 3*n
            // words are ALL priced before the first newly allocated n words.
            if fail_at > 0 {
                assert_eq!(audit.first_live, 7 * 3 * 8);
            }
            assert_audit(audit);
            assert_eq!(owner.cursors(), cursors);
            assert_eq!(usage.live_bytes(), 3 * 8); // original sent opening
            drop(messages);
            assert_eq!(usage.live_bytes(), 0);
        }
    }
}

fn compound_owners(limits: PrssLimits) -> [multiply::PartyRuntime; 3] {
    compound_setup(limits).0
}

fn compound_setup(limits: PrssLimits) -> ([multiply::PartyRuntime; 3], Party) {
    let (context, _) = setup();
    let mut setups = [Party::P0, Party::P1, Party::P2]
        .map(|p| multiply::SetupParty::new(context, p, limits).unwrap());
    let public = setups.each_mut().map(|p| p.contribution().unwrap());
    for (i, owner) in setups.iter_mut().enumerate() {
        for (j, message) in public.iter().enumerate() {
            if i != j {
                owner.receive_public(*message).unwrap();
            }
        }
    }
    let mut seeds = setups.each_mut().map(|p| Some(p.share_seed().unwrap()));
    for (i, owner) in setups.iter_mut().enumerate() {
        owner
            .receive_seed(seeds[(i + 1) % 3].take().unwrap())
            .unwrap();
    }
    // Test-only public prediction permits deterministic role-specific failure
    // injection. The runtime must wait until after rotation to draw its pivot.
    use aes::cipher::{BlockEncrypt, KeyInit};
    let seed = public
        .iter()
        .fold(0_u128, |sum, p| sum.wrapping_add(p.value()));
    let cipher = aes::Aes128::new_from_slice(&seed.to_le_bytes()).unwrap();
    let mut block = aes::cipher::Block::<aes::Aes128>::default();
    cipher.encrypt_block(&mut block);
    let raw = u64::from_le_bytes(block[..8].try_into().unwrap());
    let pivot = [Party::P0, Party::P1, Party::P2][(raw % 3) as usize];
    (setups.each_mut().map(|p| p.finish().unwrap()), pivot)
}

fn compound_bounds() -> [multiply::InputBounds; 3] {
    [multiply::InputBounds {
        left: multiply::Interval::new(-1 << 18, 1 << 18).unwrap(),
        right: multiply::Interval::new(-1 << 18, 1 << 18).unwrap(),
    }; 3]
}

fn compound_inputs(op: multiply::Plan, party: Party) -> (ReplicatedShare, ReplicatedShare) {
    // Public fixture pairs for zero, with nonzero local shares.
    let [left, right] = [multiply::Operand::Left, multiply::Operand::Right].map(|operand| {
        let n = op.operand_binding(operand).lanes();
        let parts = [
            vec![17_u64; n],
            vec![29_u64; n],
            vec![0_u64.wrapping_sub(46); n],
        ];
        op.share_operand(
            operand,
            party,
            &parts[party as usize],
            &parts[party.next() as usize],
            Limits::default(),
        )
        .unwrap()
    });
    (left, right)
}

fn compound_messages(owners: &mut [multiply::PartyRuntime; 3], op: multiply::Plan) -> Vec<Message> {
    owners
        .iter_mut()
        .map(|owner| {
            let (x, y) = compound_inputs(op, owner.party());
            owner.begin(x, y).unwrap()
        })
        .collect()
}

#[test]
fn compound_invalid_interval_and_work_reject_before_allocation_or_randomness() {
    let mut owners = compound_owners(PrssLimits::default());
    let invalid = [multiply::InputBounds {
        left: multiply::Interval::new(1 << 32, 1 << 32).unwrap(),
        right: multiply::Interval::new(1 << 32, 1 << 32).unwrap(),
    }];
    let (result, bytes) = measured(|| owners[0].plan(&invalid));
    assert_eq!(result, Err(Error::Bounds));
    assert_eq!(bytes, 0);
    assert_eq!(owners[0].cursors(), Default::default());
    let mut owners = compound_owners(PrssLimits::new(Limits::default(), 1).unwrap());
    let (result, bytes) = measured(|| owners[0].plan(&compound_bounds()));
    assert_eq!(result, Err(Error::Bounds));
    assert_eq!(bytes, 0);
    assert_eq!(owners[0].cursors(), Default::default());
}

#[test]
fn compound_multiplication_allocation_sweep_erases_before_refund() {
    for kind in 0..3 {
        let matrix = kind == 1;
        for role in 0..3 {
            for fail_at in 0..=3 {
                let mut owners = compound_owners(PrssLimits::default());
                let owner = &mut owners[role];
                let op = if matrix {
                    owner
                        .plan_matmul(
                            multiply::MatrixShape::new(3, 2, 1).unwrap(),
                            compound_bounds()[0],
                        )
                        .unwrap()
                } else if kind == 2 {
                    owner.plan_ring_product(&compound_bounds()).unwrap()
                } else {
                    owner.plan(&compound_bounds()).unwrap()
                };
                let (x, y) = compound_inputs(op, owner.party());
                let usage = owner.payload_usage();
                let watch = Watch::start(&usage, true, false);
                FAIL_AT.set(Some(fail_at));
                let result = owner.begin(x, y);
                FAIL_AT.set(None);
                if fail_at < 3 {
                    assert!(matches!(result, Err(Error::Allocation)));
                    assert_eq!(owner.status(), multiply::Status::Retired);
                } else {
                    assert!(result.is_ok());
                }
                owner.cancel();
                assert_eq!(usage.live_bytes(), if fail_at == 3 { 3 * 8 } else { 0 });
                drop(result);
                let audit = watch.finish();
                assert_eq!(audit.allocations, fail_at.min(3));
                if fail_at > 0 {
                    assert_eq!(
                        audit.first_live,
                        if matrix {
                            (2 * 6 + 2 * 2 + 3) * 8
                        } else {
                            5 * 3 * 8
                        }
                    );
                }
                assert_audit(audit);
                assert_eq!(usage.live_bytes(), 0);
                assert_eq!(owner.cursors().pair_blocks, 2);
                assert!(owner.plan(&compound_bounds()).is_err());
            }
        }
    }
}

#[test]
fn compound_rotation_allocation_sweep_retires_product_and_truncation_together() {
    for role in 0..3 {
        let allocations = if role == 2 { 6 } else { 4 };
        for fail_at in 0..=allocations {
            let (mut owners, pivot) = compound_setup(PrssLimits::default());
            let ops = owners
                .each_mut()
                .map(|p| p.plan(&compound_bounds()).unwrap());
            assert_eq!(ops, [ops[0]; 3]);
            let op = ops[0];
            let mut messages = compound_messages(&mut owners, op);
            let party = [pivot, pivot.next(), pivot.next().next()][role];
            let index = messages
                .iter()
                .position(|m| m.header().to == party)
                .unwrap();
            let next = messages.swap_remove(index);
            let owner = &mut owners[party as usize];
            let usage = owner.payload_usage();
            let watch = Watch::start(&usage, true, false);
            FAIL_AT.set(Some(fail_at));
            let result = owner.rotate(next);
            FAIL_AT.set(None);
            if fail_at < allocations {
                assert!(matches!(result, Err(Error::Allocation)));
                assert_eq!(owner.status(), multiply::Status::Retired);
            } else {
                assert!(result.is_ok());
            }
            drop(result);
            owner.cancel();
            let audit = watch.finish();
            assert_eq!(audit.allocations, fail_at.min(allocations));
            if fail_at > 0 {
                assert_eq!(audit.first_live, 4 * 3 * 8);
            }
            assert_audit(audit);
            assert_eq!(usage.live_bytes(), 3 * 8); // original outgoing rotation
            assert_eq!(owner.cursors().pair_blocks, 6);
            assert_eq!(
                owner.cursors().private_blocks,
                if role == 2 { 3 } else { 0 }
            );
            drop(messages);
            assert_eq!(usage.live_bytes(), 0);
            assert!(owner.plan(&compound_bounds()).is_err());
        }
    }
}

#[test]
fn compound_original_rotation_payload_survives_owner_and_is_erased_before_refund() {
    let mut owners = compound_owners(PrssLimits::default());
    let owner = &mut owners[0];
    let op = owner.plan(&compound_bounds()).unwrap();
    let (x, y) = compound_inputs(op, owner.party());
    let usage = owner.payload_usage();
    let message = owner.begin(x, y).unwrap();
    let watch = Watch::start(&usage, false, true);
    let mut audit = AUDIT.get();
    audit.add(message.words().as_ptr() as usize, message.words().len() * 8);
    AUDIT.set(audit);
    drop(owners);
    assert_eq!(usage.live_bytes(), 3 * 8);
    drop(message);
    assert_eq!(usage.live_bytes(), 0);
    assert_audit(watch.finish());
}

#[test]
fn boolean_allocation_failures_and_original_frames_keep_the_common_reservation() {
    for bits in [1, 9, 17, 64] {
        for role in 0..3 {
            for fail_at in 0..=3 {
                let mut owners = compound_owners(PrssLimits::default());
                let owner = &mut owners[role];
                let plan = owner.plan_and(3, bits, bits).unwrap();
                let [left, right] =
                    [multiply::Operand::Left, multiply::Operand::Right].map(|operand| {
                        plan.share_operand(
                            operand,
                            owner.party(),
                            &[17; 3],
                            &[29; 3],
                            Limits::default(),
                        )
                        .unwrap()
                    });
                let usage = owner.payload_usage();
                let watch = Watch::start(&usage, true, false);
                FAIL_AT.set(Some(fail_at));
                let result = owner.begin_and(left, right);
                FAIL_AT.set(None);
                if fail_at < 3 {
                    assert!(matches!(result, Err(Error::Allocation)));
                    assert_eq!(owner.status(), multiply::Status::Retired);
                } else {
                    assert!(result.is_ok());
                }
                owner.cancel();
                let remaining = if fail_at == 3 { 3 * 8 } else { 0 };
                assert_eq!(usage.live_bytes(), remaining);
                drop(result);
                let audit = watch.finish();
                assert_eq!(audit.allocations, fail_at.min(3));
                if fail_at > 0 {
                    assert_eq!(audit.first_live, 5 * 3 * 8);
                }
                assert_audit(audit);
                assert_eq!(usage.live_bytes(), 0);
                let q = (3 * plan.backing_bytes() as u64).div_ceil(16);
                assert_eq!(owner.cursors().pair_blocks, q);
                assert_eq!(owner.cursors().aes_work, 2 * q);
                assert_eq!(owner.cursors().public_blocks, 0);
                assert_eq!(owner.cursors().private_blocks, 0);
            }
        }
    }
}

#[test]
fn conversion_allocation_failure_sweep_covers_every_role_and_circuit_phase() {
    use multiply::conversion::Kind;
    for kind in [Kind::Bits, Kind::MostSignificantBit] {
        for role in 0..3 {
            for phase in 0..=8 {
                let allocations = match phase {
                    0 => 3,
                    8 => 0,
                    _ => 7,
                };
                for fail_at in 0..=allocations {
                    let mut owners = compound_owners(PrssLimits::default());
                    let plans = owners
                        .each_mut()
                        .map(|p| p.plan_conversion(3, kind).unwrap());
                    let mut inputs = owners.each_ref().map(|p| {
                        Some(
                            plans[0]
                                .share_input(
                                    p.party(),
                                    &[123, 456, 789],
                                    &[123, 456, 789],
                                    Limits::default(),
                                )
                                .unwrap(),
                        )
                    });
                    let mut messages: [Option<Message>; 3] = [None, None, None];
                    if phase > 0 {
                        for i in 0..3 {
                            messages[i] = Some(
                                owners[i]
                                    .begin_conversion(inputs[i].take().unwrap())
                                    .unwrap(),
                            );
                        }
                        for _ in 1..phase {
                            let mut next: [Option<Message>; 3] = [None, None, None];
                            for i in 0..3 {
                                next[i] = owners[i]
                                    .advance_conversion(messages[(i + 1) % 3].take().unwrap())
                                    .unwrap();
                            }
                            messages = next;
                        }
                    }
                    let owner = &mut owners[role];
                    let usage = owner.payload_usage();
                    let watch = Watch::start(&usage, true, false);
                    FAIL_AT.set(Some(fail_at));
                    let result = if phase == 0 {
                        owner
                            .begin_conversion(inputs[role].take().unwrap())
                            .map(Some)
                    } else {
                        owner.advance_conversion(messages[(role + 1) % 3].take().unwrap())
                    };
                    FAIL_AT.set(None);
                    if fail_at < allocations {
                        assert!(matches!(result, Err(Error::Allocation)));
                        assert_eq!(owner.status(), multiply::Status::Retired);
                    } else {
                        assert!(result.is_ok());
                    }
                    owner.cancel();
                    drop(result);
                    assert_audit(watch.finish());
                    for p in &mut owners {
                        p.cancel();
                    }
                    drop(messages);
                    drop(inputs);
                    assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
                }
            }
        }
    }
}

fn tensor_owners(
    kind: multiply::tensor::Activation,
    lanes: usize,
) -> [multiply::tensor::Executor; 3] {
    use multiply::{
        tensor::{Executor, Program},
        Interval,
    };
    let program =
        Program::activation(kind, lanes, Interval::new(-9 << 18, 9 << 18).unwrap()).unwrap();
    let runtimes = compound_owners(PrssLimits::default());
    let operation = Operation::new(
        runtimes[0].context(),
        0,
        Party::P0,
        lanes,
        18,
        Limits::default(),
    )
    .unwrap();
    let parts = [
        vec![7; lanes],
        vec![u64::MAX; lanes],
        vec![0_u64.wrapping_sub(6); lanes],
    ];
    runtimes.map(|runtime| {
        let p = runtime.party() as usize;
        let input = ReplicatedShare::from_local(
            operation,
            runtime.party(),
            &parts[p],
            &parts[(p + 1) % 3],
            Limits::default(),
        )
        .unwrap();
        Executor::new(runtime, program.clone(), input).unwrap()
    })
}

fn tensor_run(
    owners: &mut [multiply::tensor::Executor; 3],
    watched: Option<usize>,
) -> Result<(), Error> {
    use multiply::tensor::Frame;
    let collect = |party| {
        let mut a = AUDIT.get();
        a.collect = watched == Some(party);
        AUDIT.set(a);
    };
    while !owners[0].complete() {
        let mut messages: [Option<Frame>; 9] = std::array::from_fn(|_| None);
        let mut count = 0;
        for (p, owner) in owners.iter_mut().enumerate() {
            collect(p);
            let result = owner.start_step();
            collect(usize::MAX);
            for frame in result?.into_iter().flatten() {
                messages[count] = Some(frame);
                count += 1;
            }
        }
        while count > 0 {
            let mut next: [Option<Frame>; 9] = std::array::from_fn(|_| None);
            count = 0;
            for frame in messages.into_iter().flatten() {
                let to = frame.recipient() as usize;
                collect(to);
                let result = owners[to].receive(frame);
                collect(usize::MAX);
                for frame in result?.into_iter().flatten() {
                    next[count] = Some(frame);
                    count += 1;
                }
            }
            messages = next;
        }
        for owner in owners.iter_mut() {
            owner.finish_step()?;
        }
    }
    Ok(())
}

#[test]
fn tensor_allocation_failure_sweep_burns_complete_activation_owners() {
    use multiply::tensor::Activation;
    for kind in [Activation::Gelu, Activation::Silu] {
        let mut completed = false;
        for fail_at in 0..3000 {
            let mut owners = tensor_owners(kind, 3);
            let usages = owners.each_ref().map(|o| o.payload_usage());
            FAIL_AT.set(Some(fail_at));
            let result = tensor_run(&mut owners, None);
            FAIL_AT.set(None);
            if result.is_err() {
                assert_eq!(result, Err(Error::Allocation));
                assert!(owners.iter().any(|o| o.retired()));
            } else {
                completed = true;
            }
            for owner in &mut owners {
                owner.cancel();
            }
            drop(owners);
            assert!(usages.iter().all(|u| u.live_bytes() == 0));
            if completed {
                assert!(fail_at > 1000);
                break;
            }
        }
        assert!(
            completed,
            "allocation sweep did not reach complete execution"
        );
    }
}

#[test]
fn tensor_registers_and_frames_are_zeroized_before_live_charge_refund() {
    use multiply::tensor::{Activation, MAX_TILE_LANES};
    for kind in [Activation::Gelu, Activation::Silu] {
        for p in 0..3 {
            let mut owners = tensor_owners(kind, MAX_TILE_LANES);
            let usages = owners.each_ref().map(|o| o.payload_usage());
            let watch = Watch::start(&usages[p], false, false);
            tensor_run(&mut owners, Some(p)).unwrap();
            let outputs = owners.map(|o| o.finish().unwrap().1);
            drop(outputs);
            let audit = watch.finish();
            assert!(audit.allocations > 300);
            assert_audit(audit);
            assert!(usages.iter().all(|u| u.live_bytes() == 0));
        }
    }
}

fn boolean_inputs(
    owners: &mut [multiply::PartyRuntime; 3],
    bits: u8,
) -> [multiply::boolean::Share; 3] {
    let plans = owners
        .each_mut()
        .map(|p| p.plan_and(3, bits, bits).unwrap());
    let mut frames: [Option<Message>; 3] = [None, None, None];
    for i in 0..3 {
        let [left, right] = [multiply::Operand::Left, multiply::Operand::Right].map(|side| {
            plans[i]
                .share_operand(side, owners[i].party(), &[0; 3], &[0; 3], Limits::default())
                .unwrap()
        });
        frames[i] = Some(owners[i].begin_and(left, right).unwrap());
    }
    for i in 0..3 {
        owners[i]
            .receive_and(frames[(i + 1) % 3].take().unwrap())
            .unwrap();
    }
    owners.each_mut().map(|p| p.take_boolean_output().unwrap())
}

#[test]
fn ot_conversion_allocation_sweeps_keep_masks_and_accepted_outputs_charged() {
    for bits in [0, 1, 8] {
        for role in 0..3 {
            // The source public pivot is role-random; determine this owner's
            // role before arming the fail allocator, and test every index.
            for fail_at in 0..=5 {
                let mut owners = compound_owners(PrssLimits::default());
                let mut input = boolean_inputs(&mut owners, bits).map(Some);
                let plans = std::array::from_fn::<_, 3, _>(|i| {
                    owners[i]
                        .plan_boolean_to_arithmetic(input[i].as_ref().unwrap())
                        .unwrap()
                });
                let owner = &mut owners[role];
                let plan = plans[role];
                let allocations = if bits == 0 {
                    2
                } else if owner.party() == plan.receiver() {
                    3
                } else {
                    5
                };
                if fail_at > allocations {
                    continue;
                }
                let usage = owner.payload_usage();
                let watch = Watch::start(&usage, true, false);
                FAIL_AT.set(Some(fail_at));
                let result = owner.begin_boolean_to_arithmetic(input[role].take().unwrap());
                FAIL_AT.set(None);
                if fail_at < allocations {
                    assert!(matches!(result, Err(Error::Allocation)));
                    assert_eq!(owner.status(), multiply::Status::Retired);
                } else {
                    assert!(result.is_ok());
                }
                owner.cancel();
                drop(result);
                assert_audit(watch.finish());
                for p in &mut owners {
                    p.cancel();
                }
                drop(input);
                assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
            }
        }
        if bits == 0 {
            continue;
        }
        for fail_at in 0..=2 {
            let mut owners = compound_owners(PrssLimits::default());
            let input = boolean_inputs(&mut owners, bits);
            let plans = std::array::from_fn::<_, 3, _>(|i| {
                owners[i].plan_boolean_to_arithmetic(&input[i]).unwrap()
            });
            let plan = plans[0];
            let mut frames: Vec<_> = owners
                .iter_mut()
                .zip(input)
                .flat_map(|(p, x)| {
                    p.begin_boolean_to_arithmetic(x)
                        .unwrap()
                        .into_iter()
                        .flatten()
                })
                .collect();
            let messages = [
                MessagePhase::BooleanOtMask,
                MessagePhase::BooleanOtZero,
                MessagePhase::BooleanOtOne,
            ]
            .map(|phase| {
                let at = frames
                    .iter()
                    .position(|m| m.header().phase == phase)
                    .unwrap();
                Some(frames.remove(at))
            });
            let owner = &mut owners[plan.receiver() as usize];
            let usage = owner.payload_usage();
            let watch = Watch::start(&usage, true, false);
            FAIL_AT.set(Some(fail_at));
            let result = owner.receive_boolean_to_arithmetic(messages);
            FAIL_AT.set(None);
            if fail_at < 2 {
                assert!(matches!(result, Err(Error::Allocation)));
                assert_eq!(owner.status(), multiply::Status::Retired);
            } else {
                assert!(result.is_ok());
            }
            owner.cancel();
            drop(result);
            assert_audit(watch.finish());
            for p in &mut owners {
                p.cancel();
            }
            assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
        }
    }
}

#[test]
fn ppa_conversion_allocation_sweeps_cover_carry_reveal_and_output_rotation() {
    for role in 0..3 {
        for phase in 0..=10 {
            let allocations = match phase {
                0 => 5,
                1..=7 => 7,
                8 if role != 0 => 1,
                9 if role == 0 => 1,
                _ => 0,
            };
            for fail_at in 0..=allocations {
                let mut owners = compound_owners(PrssLimits::default());
                let mut input = boolean_inputs(&mut owners, 64).map(Some);
                for i in 0..3 {
                    owners[i]
                        .plan_boolean_to_arithmetic(input[i].as_ref().unwrap())
                        .unwrap();
                }
                let mut messages = Vec::new();
                if phase > 0 {
                    for i in 0..3 {
                        messages.extend(
                            owners[i]
                                .begin_boolean_to_arithmetic(input[i].take().unwrap())
                                .unwrap()
                                .into_iter()
                                .flatten(),
                        );
                    }
                    for _ in 1..phase.min(9) {
                        let mut next = Vec::new();
                        for message in messages.drain(..) {
                            let to = message.header().to as usize;
                            next.extend(
                                owners[to]
                                    .advance_boolean_to_arithmetic(message)
                                    .unwrap()
                                    .into_iter()
                                    .flatten(),
                            );
                        }
                        messages = next;
                    }
                    if phase >= 9 && (role != 0 || phase == 10) {
                        let at = messages
                            .iter()
                            .position(|m| m.header().phase == MessagePhase::BooleanPpaReveal)
                            .unwrap();
                        let reveal = messages.remove(at);
                        messages.extend(
                            owners[0]
                                .advance_boolean_to_arithmetic(reveal)
                                .unwrap()
                                .into_iter()
                                .flatten(),
                        );
                    }
                }
                let received = if phase == 0 {
                    None
                } else {
                    let expected = if phase == 1 {
                        MessagePhase::BooleanPpaMask
                    } else if phase <= 8 {
                        MessagePhase::BooleanAnd
                    } else if phase == 9 && role == 0 {
                        MessagePhase::BooleanPpaReveal
                    } else {
                        MessagePhase::BooleanPpaRotate
                    };
                    let at = messages
                        .iter()
                        .position(|m| {
                            m.header().to as usize == role && m.header().phase == expected
                        })
                        .unwrap();
                    Some(messages.remove(at))
                };
                let owner = &mut owners[role];
                let usage = owner.payload_usage();
                let watch = Watch::start(&usage, true, false);
                FAIL_AT.set(Some(fail_at));
                let result = if phase == 0 {
                    owner.begin_boolean_to_arithmetic(input[role].take().unwrap())
                } else {
                    owner.advance_boolean_to_arithmetic(received.unwrap())
                };
                FAIL_AT.set(None);
                if fail_at < allocations {
                    assert!(matches!(result, Err(Error::Allocation)));
                    assert_eq!(owner.status(), multiply::Status::Retired);
                } else {
                    assert!(result.is_ok(), "role {role} phase {phase}");
                }
                owner.cancel();
                drop(result);
                assert_audit(watch.finish());
                for p in &mut owners {
                    p.cancel();
                }
                drop(messages);
                drop(input);
                assert!(owners.iter().all(|p| p.payload_usage().live_bytes() == 0));
            }
        }
    }
}

#[test]
fn local_affine_and_standalone_rescale_keep_payloads_until_erasure() {
    use multiply::arithmetic::Affine;
    let bounds = multiply::InputBounds {
        left: multiply::Interval::new(0, 200).unwrap(),
        right: multiply::Interval::new(0, 0).unwrap(),
    };
    for fail_at in 0..=2 {
        let mut owners = compound_owners(PrssLimits::default());
        let owner = &mut owners[0];
        let op =
            Operation::new(owner.context(), 0, owner.party(), 3, 1, Limits::default()).unwrap();
        let mut input =
            ReplicatedShare::from_local(op, owner.party(), &[17; 3], &[34; 3], Limits::default())
                .unwrap();
        let usage = owner.payload_usage();
        let watch = Watch::start(&usage, true, false);
        FAIL_AT.set(Some(fail_at));
        let result = owner.affine(
            &mut input,
            None,
            Affine {
                left: -30952,
                right: 0,
                constant: 0,
            },
            bounds,
        );
        FAIL_AT.set(None);
        if fail_at < 2 {
            assert!(matches!(result, Err(Error::Allocation)));
            assert_eq!(owner.status(), multiply::Status::Retired);
        } else {
            assert!(result.is_ok());
        }
        owner.cancel();
        assert_eq!(
            usage.live_bytes(),
            if fail_at == 2 { 4 * 3 * 8 } else { 2 * 3 * 8 }
        );
        drop(result);
        drop(input);
        assert_audit(watch.finish());
        assert_eq!(usage.live_bytes(), 0);
        assert_eq!(owner.cursors(), Default::default());
    }
    for role in 0..3 {
        let allocations = if role == 2 { 6 } else { 4 };
        for fail_at in 0..=allocations {
            let (mut owners, pivot) = compound_setup(PrssLimits::default());
            let party = [pivot, pivot.next(), pivot.next().next()][role];
            let owner = &mut owners[party as usize];
            let op =
                Operation::new(owner.context(), 0, owner.party(), 3, 1, Limits::default()).unwrap();
            let first = (party as u64 + 1) * 17;
            let second = (party.next() as u64 + 1) * 17;
            let input = ReplicatedShare::from_local(
                op,
                party,
                &[first; 3],
                &[second; 3],
                Limits::default(),
            )
            .unwrap();
            let usage = owner.payload_usage();
            let watch = Watch::start(&usage, true, false);
            FAIL_AT.set(Some(fail_at));
            let result = owner.begin_rescale(input, 18, bounds.left);
            FAIL_AT.set(None);
            if fail_at < allocations {
                assert!(matches!(result, Err(Error::Allocation)));
                assert_eq!(owner.status(), multiply::Status::Retired);
            } else {
                assert!(result.is_ok());
            }
            owner.cancel();
            drop(result);
            assert_audit(watch.finish());
            assert_eq!(usage.live_bytes(), 0);
            assert_eq!(owner.cursors().pair_blocks, 4);
            assert_eq!(owner.cursors().public_blocks, 1);
            assert_eq!(
                owner.cursors().private_blocks,
                if role == 2 { 3 } else { 0 }
            );
        }
    }
}
