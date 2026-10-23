//! Scope/admission plus genuine original-worker rescale18 parity. Test-only
//! candidate preparation never opens production routing or qualification gate.
use super::*;
use crate::replicated_decoder::{
    prepared_tensor_map::native_attempts,
    program::manifest_prepared_fixture::Q_PROJECTION_SOURCE_KEY, ONE,
};

#[path = "body_tensor_manifest_continuation_tests.rs"]
mod manifest_continuation;

fn domain() -> Interval {
    Interval::new(-ONE, ONE).unwrap()
}

fn prepare(lanes: usize, shift: u8) -> PreparedTensorMap {
    PreparedTensorMap::rescale(lanes, domain(), shift, 0, |_| Ok(())).unwrap()
}

// Raw compiler declarations suffice to test binding without minting a fake
// original Program, schedule, payload quote, runtime or execution identity.
fn nodes(prepared: &PreparedTensorMap) -> Vec<Node> {
    let lanes = prepared.lanes();
    let full = prepared.program_at(0).unwrap().clone();
    let remainder = lanes % full.lanes();
    let tail = (remainder != 0).then(|| prepared.program_at(lanes - remainder).unwrap().clone());
    vec![
        Node {
            code: Code::Input("x".into()),
            lanes,
            bounds: domain(),
        },
        Node {
            code: Code::TensorLoop {
                input: LoopInput::Slice(0),
                program: full,
                tail,
            },
            lanes,
            bounds: prepared.output_bounds().unwrap(),
        },
    ]
}

fn bind(nodes: &[Node], prepared: &PreparedTensorMap) -> Result<Binding, Error> {
    Binding::validate(nodes, 1, WeightMode::Public, 0, 1, prepared)
}

#[test]
fn rescale18_binding_preserves_exact_full_tail_owners_geometry_ids_and_work() {
    for lanes in [1, 3, 128, 129, 257] {
        let prepared = prepare(lanes, 18);
        let original = nodes(&prepared);
        let native_before = native_attempts();
        let binding = bind(&original, &prepared).unwrap();
        assert_eq!((binding.input, binding.output, binding.end), (0, 1, 2));
        assert_eq!(binding.lanes, lanes);
        assert_eq!(binding.input_bounds, domain());
        assert_eq!(
            original[1].work_budget().unwrap(),
            prepared.work_budget().unwrap()
        );
        assert_eq!(native_attempts(), native_before);
        let Code::TensorLoop { program, tail, .. } = &original[1].code else {
            unreachable!()
        };
        let mut originals = prepared.programs();
        assert!(Arc::ptr_eq(program, originals.next().unwrap()));
        assert_eq!(tail.is_some(), lanes % 128 != 0 && lanes > 128);
        if let Some(tail) = tail {
            assert!(Arc::ptr_eq(tail, originals.next().unwrap()));
        }
        assert!(originals.next().is_none());
    }
}

#[test]
fn rescale18_binding_rejects_same_digest_different_full_or_tail_allocation() {
    for replace_tail in [false, true] {
        let prepared = prepare(129, 18);
        let other = prepare(129, 18);
        let mut original = nodes(&prepared);
        let Code::TensorLoop { program, tail, .. } = &mut original[1].code else {
            unreachable!()
        };
        if replace_tail {
            let other = other.program_at(128).unwrap();
            assert_eq!(tail.as_ref().unwrap().digest(), other.digest());
            assert!(!Arc::ptr_eq(tail.as_ref().unwrap(), other));
            *tail = Some(other.clone());
        } else {
            let other = other.program_at(0).unwrap();
            assert_eq!(program.digest(), other.digest());
            assert!(!Arc::ptr_eq(program, other));
            *program = other.clone();
        }
        let before = native_attempts();
        assert!(matches!(bind(&original, &prepared), Err(Error::Binding)));
        assert_eq!(native_attempts(), before);
    }
}

#[test]
fn rescale18_binding_rejects_other_factories_and_malformed_contracts_without_fallback() {
    for shift in [1, 17, 19] {
        let prepared = prepare(3, shift);
        assert!(matches!(
            bind(&nodes(&prepared), &prepared),
            Err(Error::Binding)
        ));
    }
    let silu = PreparedTensorMap::guarded_silu(3, domain(), 0, |_| Ok(())).unwrap();
    assert!(matches!(bind(&nodes(&silu), &silu), Err(Error::Binding)));
    let prepared = prepare(129, 18);
    let original = nodes(&prepared);
    for case in 0..9 {
        let mut bad = original.clone();
        match case {
            0 => bad[0].lanes = 128,
            1 => bad[1].lanes = 128,
            2 => bad[0].bounds = Interval::new(-2 * ONE, ONE).unwrap(),
            3 => bad[1].bounds = domain(),
            4 => bad[1].code = Code::PublicValues(vec![0]),
            5 => {
                let Code::TensorLoop { input, .. } = &mut bad[1].code else {
                    unreachable!()
                };
                *input = LoopInput::TokenOffsets { owner: 0, index: 0 };
            }
            6 => {
                let Code::TensorLoop { input, .. } = &mut bad[1].code else {
                    unreachable!()
                };
                *input = LoopInput::Slice(1);
            }
            7 => {
                let Code::TensorLoop { tail, .. } = &mut bad[1].code else {
                    unreachable!()
                };
                *tail = None;
            }
            8 => {
                bad[0].code = Code::PublicInput {
                    key: "file".into(),
                    digest: [0; 32],
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                }
            }
            _ => unreachable!(),
        }
        assert!(bind(&bad, &prepared).is_err(), "case {case}");
    }
    assert!(Binding::validate(&original, 1, WeightMode::Private, 0, 1, &prepared).is_err());
    assert!(Binding::validate(&original, 2, WeightMode::Public, 0, 1, &prepared).is_err());
    assert!(Binding::validate(&original, 1, WeightMode::Public, 1, 1, &prepared).is_err());
    assert!(Binding::validate(&original, 1, WeightMode::Public, 0, 2, &prepared).is_err());
    assert!(PreparedTensorMap::rescale(0, domain(), 18, 0, |_| Ok(())).is_err());
    assert!(
        PreparedTensorMap::rescale(super::super::MAX_VALUES + 1, domain(), 18, 0, |_| Ok(()))
            .is_err()
    );
}

#[test]
fn original_worker_qualification_gate_remains_explicit_and_fail_closed() {
    assert!(matches!(
        worker_parity_gate(),
        Err(PreparationError::UnqualifiedOriginalWorkerParity)
    ));
}

#[test]
fn finalization_preadmits_retained_baseline_exact_one_short_and_overflow() {
    for lanes in [3, 128, 129, 257] {
        for mode in 0..3 {
            let prepared = prepare(lanes, 18);
            let original = nodes(&prepared);
            let work = prepared.work_budget().unwrap();
            let full = Arc::downgrade(prepared.program_at(0).unwrap());
            let binding = bind(&original, &prepared).unwrap();
            let baseline = prepared.baseline_bytes().unwrap();
            let native_before = native_attempts();
            let (builder, output) = stage(binding, prepared, baseline).unwrap();
            let quote = builder
                .proof_finish_quote(std::slice::from_ref(&output), stage_slots().unwrap())
                .unwrap();
            let required = baseline + quote;
            let before = FINISH_ENTRIES.with(std::cell::Cell::get);
            let (baseline, limit) = match mode {
                0 => (baseline, required),
                1 => (baseline, required - 1),
                _ => (usize::MAX, super::super::MAX_PROGRAM_BYTES),
            };
            let result = finish_stage(builder, output, baseline, limit);
            if mode == 0 {
                let (plan, construction) = result.unwrap();
                assert_eq!(construction, quote);
                assert_eq!(plan.work_budget(), work);
                assert_eq!(plan.statistics().semantic_sites, 1);
                assert_eq!(plan.exported_lanes(), lanes);
                assert_eq!(FINISH_ENTRIES.with(std::cell::Cell::get), before + 1);
                assert_eq!(full.strong_count(), 2);
                drop(plan);
            } else {
                assert!(matches!(result, Err(Error::Bounds)));
                assert_eq!(FINISH_ENTRIES.with(std::cell::Cell::get), before);
            }
            assert_eq!(full.strong_count(), 1);
            assert_eq!(native_attempts(), native_before);
        }
    }
}

#[test]
fn actual_plan_execution_and_restoration_pressure_adds_original_baseline() {
    let prepared = prepare(129, 18);
    let original = nodes(&prepared);
    let binding = bind(&original, &prepared).unwrap();
    let (builder, output) = stage(binding, prepared, 0).unwrap();
    let (plan, construction) =
        finish_stage(builder, output, 0, super::super::MAX_PROGRAM_BYTES).unwrap();
    let mut phases = Phases::new(0, construction, 129, &plan).unwrap();
    assert!(phases.restoration > phases.execution);
    let peak = phases.construction.max(phases.restoration);
    phases.baseline = super::super::MAX_PROGRAM_BYTES - peak;
    phases.admit().unwrap();
    phases.baseline += 1;
    assert!(matches!(phases.admit(), Err(Error::Bounds)));
    assert!(matches!(
        Phases::new(usize::MAX, construction, 129, &plan),
        Err(Error::Bounds)
    ));
}

#[test]
fn staged_metadata_uses_original_capability_and_rejects_retained_pressure_before_allocation() {
    for pressure in [false, true] {
        let prepared = prepare(129, 18);
        let original = nodes(&prepared);
        let binding = bind(&original, &prepared).unwrap();
        let full = Arc::downgrade(prepared.program_at(0).unwrap());
        let tail = Arc::downgrade(prepared.program_at(128).unwrap());
        let before = CONSTRUCTOR_ENTRIES.with(std::cell::Cell::get);
        let native_before = native_attempts();
        let baseline = if pressure {
            super::super::MAX_PROGRAM_BYTES
        } else {
            prepared.baseline_bytes().unwrap()
        };
        // Test the bounded metadata stage only, not the gated Compiled route.
        // No Plan, Executor, context, runtime or fake global-drain token exists.
        let result = stage(binding, prepared, baseline);
        if pressure {
            assert!(matches!(result, Err(Error::Bounds)));
            assert_eq!(CONSTRUCTOR_ENTRIES.with(std::cell::Cell::get), before);
        } else {
            let (builder, output) = result.unwrap();
            assert_eq!(output.len(), 129);
            assert_eq!(
                output.row_key(129),
                Some(layer::RowKey {
                    owner: 1,
                    offset: 0,
                    width: 129
                })
            );
            assert_eq!((full.strong_count(), tail.strong_count()), (2, 2));
            drop((builder, output));
            assert_eq!((full.strong_count(), tail.strong_count()), (1, 1));
        }
        assert_eq!(native_attempts(), native_before);
    }
}

#[test]
fn aggregate_phase_gate_adds_original_baseline_to_phase_maximum_without_template_credit() {
    let cap = super::super::MAX_PROGRAM_BYTES;
    let mut phases = Phases {
        baseline: cap - 13,
        construction: 7,
        copies: 11,
        execution: 13,
        restoration: 12,
    };
    phases.admit().unwrap();
    phases.restoration = 14;
    assert!(matches!(phases.admit(), Err(Error::Bounds)));
    phases.baseline = usize::MAX;
    assert!(matches!(phases.admit(), Err(Error::Bounds)));
}

type Frame = (puma::Party, puma::Party, PeerBytes);
type Wire = Vec<(puma::Party, puma::Party, Vec<u8>)>;

// Separate fresh roots for each comparison, same PUBLIC insecure-test factory.
// No consuming runtime/share owner is cloned, reset or extracted for inspection.
fn fresh_workers(program: Arc<Program>) -> [Worker; 3] {
    fresh_workers_with_first(program, None)
}

fn fresh_workers_with_first(program: Arc<Program>, first_clear: Option<i64>) -> [Worker; 3] {
    let context = puma::Context::new([42; 32], program.digest(), 0).unwrap();
    let roots = puma::prss::insecure_test_support::fresh_compound_parties(
        context,
        program.work_budget().limits().unwrap(),
    )
    .unwrap();
    let mut roles = roots.map(|r| Worker::new(program.clone(), r, [42; 32]).unwrap());
    for (key, n) in program.inputs() {
        let first: Vec<_> = (0..n)
            .map(|i| 0xc751f00d12345678u64.wrapping_add(i as u64 * 17))
            .collect();
        let second: Vec<_> = (0..n)
            .map(|i| 0x254277811928f93du64.wrapping_add(i as u64 * 311))
            .collect();
        let third: Vec<_> = (0..n)
            .map(|i| {
                let clear = if i == 0 {
                    first_clear.unwrap_or(-3 * (ONE / 4))
                } else {
                    (i as i64 % 7 - 3) * (ONE / 4)
                };
                (clear as u64)
                    .wrapping_sub(first[i])
                    .wrapping_sub(second[i])
            })
            .collect();
        let shares = [first, second, third];
        for (i, role) in roles.iter_mut().enumerate() {
            role.import(key, &shares[i], &shares[(i + 1) % 3]).unwrap();
        }
    }
    roles
}

type SnapshotShares = Vec<Option<(zeroize::Zeroizing<Vec<u64>>, zeroize::Zeroizing<Vec<u64>>)>>;

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    step: usize,
    cursors: puma::prss::Cursors,
    shares: SnapshotShares,
    state: (usize, bool, usize, bool, bool, Vec<bool>),
}
fn snapshot(role: &Worker) -> Snapshot {
    let (step, cursors, shares) = role.test_boundary_state().unwrap();
    Snapshot {
        step,
        cursors,
        shares: shares
            .map(|value| {
                value.map(|(a, b)| {
                    (
                        zeroize::Zeroizing::new(a.to_vec()),
                        zeroize::Zeroizing::new(b.to_vec()),
                    )
                })
            })
            .collect(),
        state: body::integration::restored_state(role),
    }
}

trait Role {
    fn party(&self) -> puma::Party;
    fn complete(&self) -> bool;
    fn advance(&mut self) -> Result<Vec<(puma::Party, PeerBytes)>, Error>;
    fn receive(
        &mut self,
        raw: &[u8],
        from: puma::Party,
    ) -> Result<Vec<(puma::Party, PeerBytes)>, Error>;
}
macro_rules! role {
    ($ty:ty) => {
        impl Role for $ty {
            fn party(&self) -> puma::Party {
                <$ty>::party(self)
            }
            fn complete(&self) -> bool {
                <$ty>::complete(self)
            }
            fn advance(&mut self) -> Result<Vec<(puma::Party, PeerBytes)>, Error> {
                <$ty>::advance(self)
            }
            fn receive(
                &mut self,
                raw: &[u8],
                from: puma::Party,
            ) -> Result<Vec<(puma::Party, PeerBytes)>, Error> {
                <$ty>::receive(self, raw, from)
            }
        }
    };
}
role!(Worker);
role!(Invocation);

// Semantic executor deliberately has no public party accessor. Keep the party
// from its consumed original runtime in this constructor-only router harness.
impl Role for (puma::Party, Executor) {
    fn party(&self) -> puma::Party {
        self.0
    }
    fn complete(&self) -> bool {
        self.1.complete()
    }
    fn advance(&mut self) -> Result<Vec<(puma::Party, PeerBytes)>, Error> {
        self.1.advance()
    }
    fn receive(
        &mut self,
        raw: &[u8],
        from: puma::Party,
    ) -> Result<Vec<(puma::Party, PeerBytes)>, Error> {
        self.1.receive(raw, from)
    }
}

// Test-only in-process counterpart of the EXISTING three-role coordinator. It
// owns all roles and the router queue, never accepts a claimed drained bool, and
// alone issues capabilities in these tests. Production has no issuer: the real
// party-local Python coordinator must first gain an authenticated attempt/boundary
// transition after drain_peer_waves and all-role readiness (not local complete).
struct TestCoordinator<T> {
    roles: [T; 3],
    queue: Vec<Frame>,
}
impl<T: Role> TestCoordinator<T> {
    fn check_router(&self) -> Result<(), Error> {
        if !self.queue.is_empty() {
            return Err(Error::Phase);
        }
        if self
            .roles
            .iter()
            .enumerate()
            .any(|(i, role)| role.party() as usize != i)
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
}
impl TestCoordinator<Worker> {
    fn certify_start(self, compiled: &Compiled) -> Result<[CoordinatorDrained<Worker>; 3], Error> {
        self.check_router()?;
        for role in &self.roles {
            SuspendedWorker::check_start(role, &compiled.program, compiled.binding.output)?;
        }
        Ok(self.roles.map(|owner| CoordinatorDrained { owner }))
    }

    fn begin(self, compiled: &Arc<Compiled>) -> Result<[Invocation; 3], Error> {
        let [a, b, c] = self.certify_start(compiled)?;
        Ok([
            compiled.clone().begin(a)?,
            compiled.clone().begin(b)?,
            compiled.clone().begin(c)?,
        ])
    }
}
impl TestCoordinator<Invocation> {
    fn certify_end(self) -> Result<[CoordinatorDrained<Invocation>; 3], Error> {
        self.check_router()?;
        let compiled = &self.roles[0].compiled;
        for role in &self.roles {
            if !Arc::ptr_eq(compiled, &role.compiled) {
                return Err(Error::Binding);
            }
            if role.executor.is_none() {
                return Err(Error::Retired);
            }
            if !role.complete() {
                return Err(Error::Phase);
            }
        }
        Ok(self.roles.map(|owner| CoordinatorDrained { owner }))
    }

    fn restore(self) -> Result<[Worker; 3], Error> {
        let [a, b, c] = self.certify_end()?;
        Ok([a.restore()?, b.restore()?, c.restore()?])
    }
}

// Called only immediately after reach/drive drained the real router. Retained
// delivered buffers remain ordinary caller owners, not queued/in-flight work.
fn begin_after_reach(compiled: &Arc<Compiled>, roles: [Worker; 3]) -> [Invocation; 3] {
    TestCoordinator {
        roles,
        queue: Vec::new(),
    }
    .begin(compiled)
    .unwrap()
}
fn restore_after_drive(roles: [Invocation; 3]) -> Result<[Worker; 3], Error> {
    TestCoordinator {
        roles,
        queue: Vec::new(),
    }
    .restore()
}

// Router owns real frames, including their original native ledger reservations.
// Keep delivered PeerBytes alive over completion/restoration as caller owners;
// these external router buffers are NOT discounted into native admission.
fn drain<T: Role>(roles: &mut [T; 3], mut queue: Vec<Frame>, wire: &mut Wire) -> Vec<PeerBytes> {
    let mut retained = Vec::new();
    for _ in 0..100_000 {
        if queue.is_empty() {
            return retained;
        }
        for (from, to, raw) in std::mem::take(&mut queue) {
            wire.push((from, to, raw.as_bytes().to_vec()));
            queue.extend(
                roles[to as usize]
                    .receive(raw.as_bytes(), from)
                    .unwrap()
                    .into_iter()
                    .map(|(next, frame)| (to, next, frame)),
            );
            retained.push(raw);
        }
    }
    panic!("router failed to drain");
}
fn start<T: Role>(roles: &mut [T; 3]) -> Vec<Frame> {
    let mut queue = Vec::new();
    for role in roles {
        if !role.complete() {
            let from = role.party();
            queue.extend(
                role.advance()
                    .unwrap()
                    .into_iter()
                    .map(|(to, raw)| (from, to, raw)),
            );
        }
    }
    queue
}
fn drive<T: Role>(roles: &mut [T; 3]) -> (Wire, Vec<PeerBytes>) {
    let mut wire = Vec::new();
    let mut retained = Vec::new();
    for _ in 0..100_000 {
        if roles.iter().all(Role::complete) {
            return (wire, retained);
        }
        let frames = start(roles);
        retained.extend(drain(roles, frames, &mut wire));
    }
    panic!("roles failed to complete");
}
fn reach(roles: &mut [Worker; 3], target: usize, compiled: Option<&Compiled>) -> Wire {
    let mut wire = Vec::new();
    for _ in 0..100_000 {
        let mut queue = Vec::new();
        let mut reached = 0;
        for role in roles.iter_mut() {
            let from = role.party();
            let progress = match compiled {
                Some(c) => c.advance_to_start(role),
                None => role.advance_before(target),
            }
            .unwrap();
            match progress {
                BoundaryProgress::Reached => reached += 1,
                BoundaryProgress::Frames(frames) => {
                    queue.extend(frames.into_iter().map(|(to, raw)| (from, to, raw)))
                }
            }
        }
        if reached == 3 {
            assert!(queue.is_empty());
            return wire;
        }
        drop(drain(roles, queue, &mut wire));
    }
    panic!("all-role boundary not reached");
}

#[test]
fn genuine_original_worker_full_tail_wire_components_cursors_work_retirement_and_suffix_parity() {
    run_large_stack(worker_parity);
}

fn run_large_stack(run: fn()) {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}

fn worker_parity() {
    for lanes in [1, 128, 129, 257] {
        let (program, prepared, input, output) =
            super::super::program::prepared_rescale18_worker_fixture(lanes, domain()).unwrap();
        let native_before = native_attempts();
        let work = prepared.work_budget().unwrap();
        let compiled = Compiled::prepare_candidate(
            program.clone(),
            WeightMode::Public,
            input,
            output,
            prepared,
        )
        .unwrap();
        assert_eq!(native_attempts(), native_before);
        assert_eq!(program.work_budget(), work); // Remaining identity is local work.
        assert_eq!(compiled.plan.work_budget(), work);
        assert_eq!(program.retire, vec![vec![], vec![0], vec![2]]);
        let mut ordinary = fresh_workers(program.clone());
        let (whole_wire, held_baseline) = drive(&mut ordinary);
        let whole_state = ordinary.each_ref().map(snapshot);
        let whole_shares = ordinary.map(|r| r.finish().unwrap().to_vec());

        let mut reference = fresh_workers(program.clone());
        let mut candidate = fresh_workers(program.clone());
        assert!(reach(&mut reference, output, None).is_empty());
        assert!(reach(&mut candidate, output, Some(&compiled)).is_empty());
        assert_eq!(
            candidate.each_ref().map(snapshot),
            reference.each_ref().map(snapshot)
        );
        let reference_wire = reach(&mut reference, output + 1, None);
        let mut bodies = begin_after_reach(&compiled, candidate);
        let (mut wire, held_body) = drive(&mut bodies);
        assert!(!wire.is_empty());
        assert_eq!(wire, reference_wire);
        // drive returns only after all roles complete AND router is empty.
        let mut resumed = restore_after_drive(bodies).unwrap();
        assert_eq!(
            resumed.each_ref().map(snapshot),
            reference.each_ref().map(snapshot)
        );
        for r in &resumed {
            let s = snapshot(r);
            assert_eq!(s.step, output + 1);
            assert!(s.shares[input].is_none());
            assert!(s.shares[output].is_some() && s.shares[output + 1].is_none());
        }
        let (suffix, held_suffix) = drive(&mut resumed);
        wire.extend(suffix);
        assert_eq!(wire, whole_wire);
        assert_eq!(resumed.each_ref().map(snapshot), whole_state);
        assert_eq!(resumed.map(|r| r.finish().unwrap().to_vec()), whole_shares);
        assert!(!held_body.is_empty());
        for (raw, (_, _, expected)) in held_body.iter().zip(&wire) {
            assert_eq!(raw.as_bytes(), expected);
        }
        drop((held_body, held_suffix, held_baseline));
    }
}

#[test]
fn genuine_original_preparation_stays_production_gated_and_tail_faults_burn() {
    run_large_stack(tail_faults);
}

fn tail_faults() {
    let (program, prepared, input, output) =
        super::super::program::prepared_rescale18_worker_fixture(129, domain()).unwrap();
    assert!(matches!(
        Compiled::prepare(program, WeightMode::Public, input, output, prepared),
        Err(PreparationError::UnqualifiedOriginalWorkerParity)
    ));
    for malformed in [false, true] {
        let (program, prepared, input, output) =
            super::super::program::prepared_rescale18_worker_fixture(129, domain()).unwrap();
        let compiled = Compiled::prepare_candidate(
            program.clone(),
            WeightMode::Public,
            input,
            output,
            prepared,
        )
        .unwrap();
        let mut roles = fresh_workers(program);
        reach(&mut roles, output, Some(&compiled));
        let mut bodies = begin_after_reach(&compiled, roles);
        let full = start(&mut bodies);
        assert!(!full.is_empty());
        let held_full = drain(&mut bodies, full, &mut Vec::new());
        // One native rescale instruction: next advance starts original tail.
        let tail = start(&mut bodies);
        assert!(!tail.is_empty());
        let bytes: Vec<_> = tail
            .iter()
            .map(|(_, _, raw)| raw.as_bytes().to_vec())
            .collect();
        for body in &mut bodies {
            assert!(!body.complete());
            if malformed {
                assert!(body.receive(b"bad", body.party().next()).is_err());
            } else {
                body.cancel();
            }
            assert!(matches!(body.advance(), Err(Error::Retired)));
            assert!(matches!(
                body.receive(b"bad", body.party().next()),
                Err(Error::Retired)
            ));
        }
        for ((_, _, raw), expected) in tail.iter().zip(bytes) {
            assert_eq!(raw.as_bytes(), expected);
        }
        assert!(matches!(restore_after_drive(bodies), Err(Error::Retired)));
        assert_eq!(Arc::strong_count(&compiled), 1);
        drop((tail, held_full));
    }
}

#[test]
fn protected_suffix_retained_input_and_sticky_mask_full_tail_original_worker_parity() {
    run_large_stack(protected_suffix_parity);
}

fn protected_suffix_parity() {
    for lanes in [1, 128, 129, 257] {
        for retain_input in [false, true] {
            // Plain protected suffix, accepted prefix check, rejected prefix
            // check. All routes/geometry are public and identical within each
            // original/candidate comparison; only imported clear values differ.
            for case in 0..3 {
                let sticky = case != 0;
                let rejected = case == 2;
                let clear = Some(if rejected { 3 * (ONE / 4) } else { 0 });
                let (program, prepared, input, output, valid) =
                    super::super::program::prepared_rescale18_protected_worker_fixture(
                        lanes,
                        domain(),
                        retain_input,
                        sticky,
                    )
                    .unwrap();
                let map_work = prepared.work_budget().unwrap();
                let compiled = Compiled::prepare_candidate(
                    program.clone(),
                    WeightMode::Public,
                    input,
                    output,
                    prepared,
                )
                .unwrap();
                assert_eq!(compiled.plan.work_budget(), map_work);
                assert_ne!(program.work_budget(), map_work);
                assert_eq!(program.retire[output].contains(&input), !retain_input);
                let mut ordinary = fresh_workers_with_first(program.clone(), clear);
                let (whole_wire, held_whole) = drive(&mut ordinary);
                let whole_state = ordinary.each_ref().map(snapshot);
                let whole_shares = ordinary.map(|r| r.finish().unwrap().to_vec());

                let mut reference = fresh_workers_with_first(program.clone(), clear);
                let mut candidate = fresh_workers_with_first(program.clone(), clear);
                let mut wire = reach(&mut candidate, output, Some(&compiled));
                assert_eq!(wire, reach(&mut reference, output, None));
                let before = candidate.each_ref().map(snapshot);
                assert_eq!(before, reference.each_ref().map(snapshot));
                let expected_body_wire = reach(&mut reference, output + 1, None);
                let mut bodies = begin_after_reach(&compiled, candidate);
                let (body_wire, held_body) = drive(&mut bodies);
                assert_eq!(body_wire, expected_body_wire);
                wire.extend(body_wire);
                let mut resumed = restore_after_drive(bodies).unwrap();
                let seam = resumed.each_ref().map(snapshot);
                assert_eq!(seam, reference.each_ref().map(snapshot));
                for (after, before) in seam.iter().zip(&before) {
                    assert_eq!(after.step, output + 1);
                    assert_eq!(after.shares[input].is_some(), retain_input);
                    assert!(after.shares[output].is_some());
                    assert!(after.shares[output + 1].is_none());
                    if let Some(valid) = valid {
                        // Actual stored validity components, not an opened bit
                        // or a newly manufactured validity/source owner.
                        assert!(before.shares[valid].is_some());
                        assert_eq!(after.shares[valid], before.shares[valid]);
                    }
                }
                let (suffix_wire, held_suffix) = drive(&mut resumed);
                assert!(!suffix_wire.is_empty()); // Protected, not identity-only.
                wire.extend(suffix_wire);
                let final_state = resumed.each_ref().map(snapshot);
                assert_eq!(wire, whole_wire);
                assert_eq!(final_state, whole_state);
                for (after, seam) in final_state.iter().zip(&seam) {
                    assert!(after.cursors.aes_work > seam.cursors.aes_work);
                    assert!(after.cursors.pair_blocks > seam.cursors.pair_blocks);
                }
                let shares = resumed.map(|r| r.finish().unwrap().to_vec());
                assert_eq!(shares, whole_shares);
                if sticky {
                    // Additional final-disclosure contract, NEVER a replacement
                    // for full wire/two-component/cursor equality above.
                    let opened = |i| shares.iter().fold(0u64, |n, s| n.wrapping_add(s[i]));
                    assert_eq!(opened(lanes), u64::from(!rejected));
                    if rejected {
                        assert!((0..lanes).all(|i| opened(i) == 0));
                    }
                }
                drop(reference);
                assert_eq!(Arc::strong_count(&compiled), 1);
                assert_eq!(Arc::strong_count(&program), 2); // Caller + compiled.
                drop((held_whole, held_body, held_suffix));
            }
        }
    }
}

#[test]
fn sticky_retained_prefix_active_tail_failure_drops_original_worker_owners() {
    run_large_stack(sticky_tail_failure);
}

fn sticky_tail_failure() {
    for malformed in [false, true] {
        let (program, prepared, input, output, valid) =
            super::super::program::prepared_rescale18_protected_worker_fixture(
                129,
                domain(),
                true,
                true,
            )
            .unwrap();
        assert!(valid.is_some());
        let compiled = Compiled::prepare_candidate(
            program.clone(),
            WeightMode::Public,
            input,
            output,
            prepared,
        )
        .unwrap();
        let mut roles = fresh_workers_with_first(program.clone(), Some(3 * (ONE / 4)));
        reach(&mut roles, output, Some(&compiled));
        let mut bodies = begin_after_reach(&compiled, roles);
        let frames = start(&mut bodies);
        let held_full = drain(&mut bodies, frames, &mut Vec::new());
        let held_tail = start(&mut bodies);
        assert!(!held_tail.is_empty());
        let bytes: Vec<_> = held_tail
            .iter()
            .map(|(_, _, p)| p.as_bytes().to_vec())
            .collect();
        for body in &mut bodies {
            if malformed {
                assert!(body.receive(b"bad", body.party().next()).is_err());
            } else {
                body.cancel();
            }
            assert!(matches!(body.advance(), Err(Error::Retired)));
        }
        assert!(matches!(restore_after_drive(bodies), Err(Error::Retired)));
        assert_eq!(Arc::strong_count(&compiled), 1);
        assert_eq!(Arc::strong_count(&program), 2);
        for ((_, _, p), original) in held_tail.iter().zip(bytes) {
            assert_eq!(p.as_bytes(), original);
        }
        drop((held_tail, held_full));
    }
}

#[test]
fn coordinator_custody_is_transparent_and_preserves_consuming_owner_drop() {
    assert_eq!(size_of::<CoordinatorDrained<Worker>>(), size_of::<Worker>());
    assert_eq!(
        size_of::<CoordinatorDrained<Invocation>>(),
        size_of::<Invocation>()
    );
    assert_eq!(
        std::mem::align_of::<CoordinatorDrained<Worker>>(),
        std::mem::align_of::<Worker>()
    );
    assert!(std::mem::needs_drop::<CoordinatorDrained<Worker>>());
    assert!(std::mem::needs_drop::<CoordinatorDrained<Invocation>>());
}

#[test]
fn coordinator_start_requires_router_drain_all_three_roles_and_consuming_custody() {
    run_large_stack(coordinator_start_failures);
}

fn coordinator_start_failures() {
    for case in 0..4 {
        let (program, prepared, input, output) =
            super::super::program::prepared_rescale18_worker_fixture(129, domain()).unwrap();
        let native_before = native_attempts();
        let compiled = Compiled::prepare_candidate(
            program.clone(),
            WeightMode::Public,
            input,
            output,
            prepared,
        )
        .unwrap();
        let mut roles = fresh_workers(program.clone());
        reach(&mut roles, output, Some(&compiled));
        assert!(roles.iter().all(|r| snapshot(r).step == output));
        let queue = if case == 0 {
            // All local starts ready does NOT prove the router is empty. Keep
            // genuine native frames, not an invented bool or byte placeholder.
            let mut sender = fresh_workers(program.clone());
            let queue = start(&mut sender);
            assert!(!queue.is_empty());
            queue
        } else {
            Vec::new()
        };
        match case {
            1 => roles[2].cancel(), // Two ready roles are insufficient.
            2 => roles.swap(0, 1),  // Exactly one authenticated owner per role.
            _ => (),
        }
        let result = TestCoordinator { roles, queue }.certify_start(&compiled);
        match case {
            0 => assert!(matches!(result, Err(Error::Phase))),
            1 => assert!(matches!(result, Err(Error::Retired))),
            2 => assert!(matches!(result, Err(Error::Binding))),
            _ => {
                // Abandoned capability drops its EXACT original worker. No
                // reusable runtime or detached permit escapes coordinator custody.
                drop(result.unwrap());
            }
        }
        assert_eq!(native_attempts(), native_before);
        assert_eq!(Arc::strong_count(&compiled), 1);
        assert_eq!(Arc::strong_count(&program), 2);
    }
}

#[test]
fn coordinator_restore_rejects_queued_replay_partial_roles_and_retirement() {
    run_large_stack(coordinator_restore_failures);
}

fn coordinator_restore_failures() {
    for case in 0..5 {
        let (program, prepared, input, output) =
            super::super::program::prepared_rescale18_worker_fixture(129, domain()).unwrap();
        let native_before = native_attempts();
        let compiled = Compiled::prepare_candidate(
            program.clone(),
            WeightMode::Public,
            input,
            output,
            prepared,
        )
        .unwrap();
        let mut roles = fresh_workers(program.clone());
        reach(&mut roles, output, Some(&compiled));
        let mut bodies = begin_after_reach(&compiled, roles);
        let mut held = if case == 2 {
            // Drained first tile, original tail still incomplete on every role.
            let frames = start(&mut bodies);
            let held = drain(&mut bodies, frames, &mut Vec::new());
            assert!(bodies.iter().all(|b| !b.complete()));
            held
        } else {
            let (_, held) = drive(&mut bodies);
            assert!(bodies.iter().all(Invocation::complete));
            held
        };
        let queue = if case == 0 {
            // A genuine delivered frame requeued for replay must still forbid
            // restoration, even though ALL local executors report complete.
            vec![(puma::Party::P0, puma::Party::P1, held.pop().unwrap())]
        } else {
            Vec::new()
        };
        let bytes: Vec<_> = held.iter().map(|p| p.as_bytes().to_vec()).collect();
        match case {
            1 => bodies[2].cancel(),
            3 => bodies.swap(0, 1),
            _ => (),
        }
        let result = TestCoordinator {
            roles: bodies,
            queue,
        }
        .certify_end();
        match case {
            0 | 2 => assert!(matches!(result, Err(Error::Phase))),
            1 => assert!(matches!(result, Err(Error::Retired))),
            3 => assert!(matches!(result, Err(Error::Binding))),
            _ => drop(result.unwrap()), // Abandon original suspended owners.
        }
        assert_eq!(native_attempts(), native_before);
        assert_eq!(Arc::strong_count(&compiled), 1);
        assert_eq!(Arc::strong_count(&program), 2);
        for (raw, expected) in held.iter().zip(bytes) {
            assert_eq!(raw.as_bytes(), expected);
        }
    }
}

// Real authenticated file owners and a Manifest::compile-produced Program. This
// fixture deliberately does NOT invent an original prepared capability for that
// Program: current compiler drops those capabilities after publishing its loops.
// Full source-backed Worker handoff parity needs a compiler-private fixture hook
// returning the ORIGINAL selected Slice/rescale18 capability alongside the SAME
// Manifest::compile result and immutable input/output IDs. A digest-matched new
// factory, raw Program literal, or registry-only compile is not that seam.
fn public_dispatch_fixture() -> (
    super::super::Manifest,
    Arc<Program>,
    Arc<PublicSource>,
    Worker,
) {
    use super::super::{ExecutionPolicy, Manifest, Publisher, RangePolicy};
    let config = br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,"vocab_size":3,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":true,"max_position_embeddings":16,"rope_theta":10000,"rms_norm_eps":0.000001}"#;
    let policy = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        max_abs_activation: 1,
        ..ExecutionPolicy::default()
    };
    let mut publisher =
        Publisher::new_with_policy(config, 1, 1, [1; 32], [2; 32], WeightMode::Public, policy)
            .unwrap();
    let shapes = publisher.shapes().to_vec();
    for (key, shape) in shapes {
        let value = if key.contains("norm") { 1f32 } else { 0f32 };
        let bytes: Vec<_> = (0..shape.iter().product::<usize>())
            .flat_map(|_| value.to_le_bytes())
            .collect();
        publisher.append(&key, 0, &bytes).unwrap();
    }
    publisher.seal(&[0]).unwrap();
    let manifest: Manifest = serde_json::from_slice(&publisher.manifest().unwrap()).unwrap();
    drop(publisher);
    let program = manifest.compile().unwrap();
    let public = Arc::new(import_public_dispatch_source(&manifest));
    // Current Worker has an independently owned, by-value PublicSource. Both
    // originals are imported ONCE before dispatch, then separately inventoried.
    let worker_source = import_public_dispatch_source(&manifest);
    let [runtime, b, c] = puma::prss::insecure_test_support::fresh_compound_parties(
        puma::Context::new([42; 32], program.digest(), 0).unwrap(),
        program.work_budget().limits().unwrap(),
    )
    .unwrap();
    drop((b, c));
    let worker =
        Worker::new_with_source(program.clone(), runtime, [42; 32], Some(worker_source)).unwrap();
    (manifest, program, public, worker)
}

fn import_public_dispatch_source(manifest: &super::super::Manifest) -> PublicSource {
    let mut importer = super::super::PublicSourceImport::new(manifest).unwrap();
    for (key, tensor) in &manifest.tensors {
        let value = tensor.lower as f32 / ONE as f32;
        assert_eq!(tensor.lower, tensor.upper);
        let bytes: Vec<_> = (0..tensor.shape.iter().product::<usize>())
            .flat_map(|_| value.to_le_bytes())
            .collect();
        importer.append(key, 0, &bytes).unwrap();
    }
    importer.finish().unwrap()
}

fn manifest_inline_baseline(manifest: &super::super::Manifest) -> usize {
    // Source/norm maps have their own ACTUAL-capacity source census. Registry
    // validation temporaries and the publisher are already dropped by this point.
    size_of::<super::super::Manifest>() + manifest.schema.capacity() + manifest.config.capacity()
}

fn dispatch_input(party: puma::Party, lanes: usize) -> Secret {
    let mut secret = Secret::new(lanes).unwrap();
    for i in 0..lanes {
        let a = 0xc751f00d12345678u64.wrapping_add(i as u64 * 17);
        let b = 0x254277811928f93du64.wrapping_add(i as u64 * 311);
        let clear = ((i as i64 % 7 - 3) * (ONE / 4)) as u64;
        let shares = [a, b, clear.wrapping_sub(a).wrapping_sub(b)];
        secret.first[i] = shares[party as usize];
        secret.second[i] = shares[(party as usize + 1) % 3];
    }
    secret
}

#[test]
fn authenticated_source_dispatch_census_rejects_pressure_and_registry_substitution_before_stage() {
    run_large_stack(public_dispatch_rejections);
}

fn public_dispatch_rejections() {
    let (mut manifest, program, public, worker) = public_dispatch_fixture();
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let outside = manifest_inline_baseline(&manifest);
    let prepared = prepare(129, 18);
    let declarations = nodes(&prepared);
    let binding = bind(&declarations, &prepared).unwrap();
    let before = CONSTRUCTOR_ENTRIES.with(std::cell::Cell::get);
    let native_before = native_attempts();
    let source =
        SourceDispatch::new(&program, &worker, public.clone(), &registry, outside).unwrap();
    assert!(Arc::ptr_eq(&source.public, &public));
    assert_eq!(Arc::strong_count(&public), 2);
    let fixed = source.baseline_bytes - outside;
    assert_eq!(
        source.worker_metadata_bytes,
        SuspendedWorker::public_metadata_bytes(&worker, &program).unwrap()
    );
    assert!(source.baseline_bytes >= outside + 2 * public.resident_metadata_bytes().unwrap());
    drop(source);
    let exact = SourceDispatch::new(
        &program,
        &worker,
        public.clone(),
        &registry,
        super::super::MAX_PROGRAM_BYTES - fixed,
    )
    .unwrap();
    assert_eq!(exact.baseline_bytes, super::super::MAX_PROGRAM_BYTES);
    let combined = add(
        retained_program_baseline(&program, &prepared).unwrap(),
        exact.baseline_bytes,
    )
    .unwrap();
    // Source-only quote fitting exactly is NOT permission to allocate a Builder:
    // original Program/prepared owners and stage/result slots remain additive.
    assert!(matches!(
        stage(binding, prepared, combined),
        Err(Error::Bounds)
    ));
    assert_eq!(CONSTRUCTOR_ENTRIES.with(std::cell::Cell::get), before);
    drop(exact);
    for outside in [super::super::MAX_PROGRAM_BYTES - fixed + 1, usize::MAX] {
        assert!(matches!(
            SourceDispatch::new(&program, &worker, public.clone(), &registry, outside),
            Err(Error::Bounds)
        ));
    }
    let source =
        SourceDispatch::new(&program, &worker, public.clone(), &registry, outside).unwrap();
    #[allow(clippy::drop_non_drop)] // End the registry borrow before original owners.
    drop(registry);
    manifest
        .tensors
        .get_mut(Q_PROJECTION_SOURCE_KEY)
        .unwrap()
        .shape
        .reserve(1024);
    let grown = SourceRegistry::from_manifest(&manifest).unwrap();
    // Same records/binding, larger REAL metadata backing: original admission is
    // not transferable to a later borrowed registry with larger capacities.
    public.check_registry(&grown).unwrap();
    assert!(matches!(
        source.check_begin(&program, &worker, &grown),
        Err(Error::Bounds)
    ));
    drop((grown, source));
    manifest
        .tensors
        .get_mut(Q_PROJECTION_SOURCE_KEY)
        .unwrap()
        .digest[0] ^= 1;
    let substituted = SourceRegistry::from_manifest(&manifest).unwrap();
    assert!(matches!(
        SourceDispatch::new(&program, &worker, public.clone(), &substituted, outside),
        Err(Error::Binding)
    ));
    #[allow(clippy::drop_non_drop)] // End the substituted registry borrow explicitly.
    drop(substituted);
    assert_eq!(Arc::strong_count(&public), 1);
    assert_eq!(CONSTRUCTOR_ENTRIES.with(std::cell::Cell::get), before);
    assert_eq!(native_attempts(), native_before);
    // Source validation cannot turn a substituted Manifest into an admitted
    // execution Program, even though descriptor validation alone can succeed.
    // This independently recompiles through the real Manifest gate; it is not
    // included in the source-only no-new-native-factories assertion above.
    assert!(manifest.compile().is_err());
}

#[test]
fn registry_dispatch_constructor_retains_original_arcs_and_matches_expanded_source_contract() {
    run_large_stack(public_dispatch_constructor_parity);
}

fn public_dispatch_constructor_parity() {
    let (mut manifest, program, public, worker) = public_dispatch_fixture();
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let source = SourceDispatch::new(
        &program,
        &worker,
        public.clone(),
        &registry,
        manifest_inline_baseline(&manifest),
    )
    .unwrap();
    let prepared = prepare(129, 18);
    let full = Arc::downgrade(prepared.program_at(0).unwrap());
    let tail = Arc::downgrade(prepared.program_at(128).unwrap());
    let declarations = nodes(&prepared);
    let binding = bind(&declarations, &prepared).unwrap();
    // Additional raw declarations are test binding inputs, not a forged Program.
    let declaration_bytes = size_of::<Vec<Node>>()
        + array::<Node>(declarations.capacity()).unwrap()
        + declarations
            .iter()
            .map(|n| {
                if let Code::Input(k) = &n.code {
                    k.capacity()
                } else {
                    0
                }
            })
            .sum::<usize>();
    let baseline = retained_program_baseline(&program, &prepared).unwrap()
        + source.baseline_bytes
        + declaration_bytes;
    let native_before = native_attempts();
    let (builder, output) = stage(binding, prepared, baseline).unwrap();
    let (plan, construction) =
        finish_stage(builder, output, baseline, super::super::MAX_PROGRAM_BYTES).unwrap();
    let phases = Phases::new(baseline, construction, 129, &plan).unwrap();
    phases.admit().unwrap();
    assert_eq!(native_attempts(), native_before);
    let original_plan = Arc::downgrade(&plan);
    let fresh = || {
        puma::prss::insecure_test_support::fresh_compound_parties(
            puma::Context::new([42; 32], plan.digest(), 0).unwrap(),
            plan.work_budget().limits().unwrap(),
        )
        .unwrap()
    };
    for exit in 0..4 {
        let [runtime, b, c] = fresh();
        drop((b, c));
        let party = runtime.party();
        let input = dispatch_input(party, 129);
        let mut owner = source
            .executor(plan.clone(), runtime, vec![input], &registry)
            .unwrap();
        assert_eq!(Arc::strong_count(&public), 3);
        assert_eq!(original_plan.strong_count(), 2);
        match exit {
            0 => assert!(matches!(owner.finish(), Err(Error::Phase))),
            1 => drop(owner),
            2 => {
                let held = owner.advance().unwrap();
                assert!(!held.is_empty());
                owner.cancel();
                assert!(matches!(owner.advance(), Err(Error::Retired)));
                drop((held, owner));
            }
            _ => {
                assert!(owner.receive(b"bad", party.next()).is_err());
                assert!(matches!(owner.advance(), Err(Error::Retired)));
                drop(owner);
            }
        }
        assert_eq!(Arc::strong_count(&public), 2);
        assert_eq!(original_plan.strong_count(), 1);
    }
    // Constructor-only parity, NOT complete compact/Worker qualification: no
    // claim that this standalone Plan is mapped into the compiled decoder.
    let mut expected = fresh().map(|runtime| {
        let party = runtime.party();
        let input = dispatch_input(party, 129);
        (
            party,
            Executor::new(
                plan.clone(),
                runtime,
                vec![input],
                Some(public.clone()),
                Some(&program),
            )
            .unwrap(),
        )
    });
    let mut actual = fresh().map(|runtime| {
        let party = runtime.party();
        let input = dispatch_input(party, 129);
        (
            party,
            source
                .executor(plan.clone(), runtime, vec![input], &registry)
                .unwrap(),
        )
    });
    assert_eq!(original_plan.strong_count(), 7);
    assert_eq!(Arc::strong_count(&public), 8); // Caller + source + six executors.
    #[allow(clippy::drop_non_drop)] // End the registry borrow before original owners.
    drop(registry);
    manifest
        .tensors
        .get_mut(Q_PROJECTION_SOURCE_KEY)
        .unwrap()
        .digest[0] ^= 1;
    let substituted = SourceRegistry::from_manifest(&manifest).unwrap();
    let [runtime, b, c] = fresh();
    drop((b, c));
    let input = dispatch_input(runtime.party(), 129);
    assert!(matches!(
        source.executor(plan.clone(), runtime, vec![input], &substituted),
        Err(Error::Binding)
    ));
    assert_eq!(original_plan.strong_count(), 7);
    assert_eq!(Arc::strong_count(&public), 8);
    #[allow(clippy::drop_non_drop)] // End the substituted registry borrow explicitly.
    drop(substituted);
    drop(manifest); // Borrowed construction authority does not escape.
    let (wire, held) = drive(&mut actual);
    let (expected_wire, expected_held) = drive(&mut expected);
    assert_eq!(wire, expected_wire);
    assert!(!wire.is_empty());
    let expected = expected.map(|(_, owner)| owner.finish().unwrap());
    let actual = actual.map(|(_, owner)| owner.finish().unwrap());
    for ((runtime, values), (expected_runtime, expected_values)) in actual.iter().zip(&expected) {
        assert_eq!(runtime.cursors(), expected_runtime.cursors());
        assert_eq!(values.output_len(0).unwrap(), 129);
        for lane in 0..129 {
            assert_eq!(
                values.word(0, lane).unwrap(),
                expected_values.word(0, lane).unwrap()
            );
        }
    }
    assert_eq!(Arc::strong_count(&public), 2);
    drop((
        actual,
        expected,
        source,
        worker,
        program,
        declarations,
        held,
        expected_held,
    ));
    assert_eq!(Arc::strong_count(&public), 1);
    assert_eq!(original_plan.strong_count(), 1);
    assert_eq!(full.strong_count(), 1);
    assert_eq!(tail.strong_count(), 1);
    drop(plan);
    assert_eq!(full.strong_count(), 0);
    assert_eq!(tail.strong_count(), 0);
}
