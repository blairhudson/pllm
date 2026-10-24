//! Independent region-side logical inventory, not an allocator/stack/RSS proof.
//! Mount beneath `nexp_region::tests`: existing helpers supply genuine owners,
//! not expected resource totals. Native child/private implementation inventories
//! remain their owners' contracts; no CPython or opaque callframe qualification.
use super::*;
use std::alloc::Layout;
use std::mem::{align_of, size_of};
use std::sync::atomic::AtomicUsize;

type C = OwnedBudgetContext;

macro_rules! slots {
    ($($ty:ty),+ $(,)?) => {
        [$(size_of::<$ty>()),+].into_iter().sum::<usize>()
    };
}

// This record is derived from State's actual retained fields, not its quote or
// ConstructionSlots. Native children keep their complete existing owner types.
#[allow(dead_code)]
struct RetainedRole {
    root: Option<RootValue<C>>,
    bindings: [Binding; 14],
    ledger: internal_use::Ledger<18>,
    role: u8,
    phase: u8,
    drelu: Option<arithmetic::PartyFor<C>>,
    select: Option<selection::PartyFor<C>>,
    tr: Option<arithmetic::PartyFor<C>>,
    low: Option<lookup::PartyFor<C>>,
    high: Option<lookup::PartyFor<C>>,
    product: Option<consuming::PartyFor<C>>,
    gap: Option<arithmetic::PartyFor<C>>,
    work: [Zeroizing<Box<[u64]>>; 5],
    h: Option<arithmetic::MaskedOutputFor<C>>,
    v: Option<selection::MaskedOutputFor<C>>,
    j1: Option<arithmetic::MaskedOutputFor<C>>,
    u0: Option<consuming::LookupValueFor<C>>,
    u1: Option<consuming::LookupValueFor<C>>,
    p: Option<consuming::ProductValueFor<C>>,
    output: Option<CommonMaskedFor<C, Ring>>,
    tables: Arc<Tables<C>>,
    local: PhantomData<Rc<()>>,
    lease: LeaseFor<C>,
}

#[allow(dead_code)]
struct RetainedSlot {
    state: Mutex<RetainedRole>,
    lease: LeaseFor<C>,
}

#[allow(dead_code)]
struct RetainedCoordination {
    control: Mutex<custody::SplitControl>,
    slots: [Arc<Slot<C>>; 2],
    original: Option<LeaseFor<C>>,
    additional: LeaseFor<C>,
}

#[allow(dead_code)]
struct RetainedTables {
    checked: tables::Verified,
    hashes: [[u8; 32]; 2],
    declaration: Declaration,
    input: InputOrigin<C>,
    lease: LeaseFor<C>,
}

// Logical Arc contract: two counters, payload alignment and tail padding.
// This intentionally makes no assertion about the standard allocator's header.
fn arc_inventory<T>() -> usize {
    Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .unwrap()
        .0
        .pad_to_align()
        .size()
}

fn query_call_slots() -> usize {
    // resources arguments/return plus actual nested resource-query results.
    // No reference to Quote.control_bytes/bootstrap_bytes/ConstructionSlots.
    slots![
        &LocalDealerFor<C>,
        &Spec,
        &Input,
        Result<Quote, BuildError>,
        [(u8, u8, Interpretation); 14],
        usize,
        arithmetic::Resources,
        arithmetic::Resources,
        arithmetic::Resources,
        selection::Resources,
        lookup::Resources,
        product::Resources,
        certified::Quote,
        Result<certified::Quote, certified::Error>,
        Result<arithmetic::Resources, Error>,
        Result<selection::Resources, Error>,
        Result<lookup::Resources, Error>,
        Result<product::Resources, Error>,
        BoundRange<LocalAssumption>,
        Result<BoundRange<LocalAssumption>, Error>,
        [usize; 8],
        WorkQuote,
    ]
}

fn native_query_callee_slots() -> usize {
    use crate::compact_dpf::gpu127;
    // Actual resource-helper geometry/selection/calculation paths. These helpers
    // do NOT construct keys. The largest path is sequential, not three heaps.
    let geometry = slots![
        arithmetic::Spec,
        Result<arithmetic::Resources, Error>,
        Result<(gpu127::Profile, u8, usize), Error>,
        Result<gpu127::Profile, gpu127::Error>,
        Result<gpu127::Profile, Error>,
        gpu127::Profile,
    ];
    let selection = slots![
        selection::Spec,
        selection::Spec,
        Result<selection::Resources, Error>,
        [usize; 5],
    ];
    let lookup = slots![
        lookup::Spec,
        Result<lookup::Resources, Error>,
        gpu127::Resources,
        selection::Resources,
        [usize; 14],
        lookup::Spec,
        lookup::Spec,
    ];
    geometry.max(selection).max(lookup)
}

fn table_preparation_call_slots() -> usize {
    // construct_with's original input/spec stay live while prepare runs.
    // The callee's private proof/hash scratch is a separate native contract.
    slots![
        Spec,
        Quote,
        Input,
        LeaseFor<C>,
        exp_floor::Workspace,
        &mut exp_floor::Workspace,
        tables::Verified,
        [[u8; 32]; 2],
        Result<(tables::Verified, [[u8; 32]; 2]), certified::Error>,
        Result<tables::Verified, tables::Error>,
        &EdgeDescriptor,
        &[i64],
        &tables::Verified,
        [&EdgeDescriptor; 18],
        Result<bounds::BindingIntervals, Error>,
        [Option<Interval>; 14],
    ]
}

fn native_issuance_call_slots() -> usize {
    // Original/new/native return slots across construct_with and make_state.
    // Child native backing is NOT inferred from these inline records.
    slots![
        Spec,
        Quote,
        [Binding; 14],
        [ConsumerUse; 18],
        [internal_use::Ledger<18>; 3],
        [Option<LeaseFor<C>>; 8],
        [LeaseFor<C>; 5],
        RootInputFor<C>,
        Result<RootInputFor<C>, Error>,
        (DealerMaskFor<C, Ring>, [CommonMaskedFor<C, Ring>; 2]),
        Result<(DealerMaskFor<C, Ring>, [CommonMaskedFor<C, Ring>; 2]), Error>,
        [Result<(arithmetic::OutputMasksFor<C>, [arithmetic::PartyFor<C>; 2]), Error>; 3],
        Result<(selection::OutputMasksFor<C>, [selection::PartyFor<C>; 2]), Error>,
        [Result<(lookup::OutputMasksFor<C>, [lookup::PartyFor<C>; 2]), Error>; 2],
        consuming::DeclarationFor<C>,
        [consuming::LookupMaskFor<C>; 2],
        Result<(consuming::ProductMaskFor<C>, [consuming::PartyFor<C>; 2]), Error>,
        [State<C>; 2],
        Result<State<C>, Error>,
        [Arc<Slot<C>>; 2],
        Arc<Shared<C>>,
        (Mask<C>, Coordinator<C>, [Role<C>; 2]),
        Result<(Mask<C>, Coordinator<C>, [Role<C>; 2]), BuildError>,
        Zeroizing<[u8; 32]>,
        Zeroizing<[u8; 8]>,
        [Zeroizing<Box<[u64]>>; 3],
        Vec<u64>,
        Box<[u64]>,
        Result<Zeroizing<Box<[u64]>>, Error>,
    ]
}

fn role_transition_call_slots() -> usize {
    // Both roles may retain a previous frame while the next result is delivered.
    // These are the actual enum/Result/argument types, not RuntimeSlots itself.
    2 * slots![
        &mut Role<C>,
        &[u8],
        &Shared<C>,
        custody::CallbackPollFor<'static, C, State<C>>,
        custody::OperationFor<'static, C, State<C>>,
        std::sync::MutexGuard<'static, State<C>>,
        Result<Body<C>, Error>,
        Result<Option<Body<C>>, Error>,
        Result<Frame<C>, Error>,
        Result<Step<C>, Error>,
        Result<Output<C>, Error>,
        Result<arithmetic::StepFor<C>, Error>,
        Result<selection::MaskedOutputFor<C>, Error>,
        Result<lookup::StepFor<C>, Error>,
        Result<product::OutputFor<C>, Error>,
        [consuming::LookupValueFor<C>; 2],
        Result<consuming::ProductValueFor<C>, Error>,
        CommonMaskedFor<C, Ring>,
        Result<EdgeDescriptor, Error>,
        Result<&'static [u8], Error>,
        Result<(), Error>,
        Arc<Slot<C>>,
        Arc<Shared<C>>,
        u64,
    ]
}

#[test]
fn independent_concrete_phase_and_retained_inventory_for_every_qualified_width() {
    assert_eq!(size_of::<State<C>>(), size_of::<RetainedRole>());
    assert_eq!(align_of::<State<C>>(), align_of::<RetainedRole>());
    assert_eq!(size_of::<Slot<C>>(), size_of::<RetainedSlot>());
    assert_eq!(size_of::<Shared<C>>(), size_of::<RetainedCoordination>());
    assert_eq!(size_of::<Tables<C>>(), size_of::<RetainedTables>());

    let root = OwnedBudget::new(64 << 20).unwrap();
    let source = source();
    let dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    for lanes in 1..=256 {
        let values = vec![0; lanes]; // Test harness only; not a region allocation.
        let input = owner(&root, &values, 17);
        let original = root.charge_external(bootstrap_bytes::<C>()).unwrap();
        let q = resources(&dealer, &spec(source, lanes), &input).unwrap();
        assert!(
            bootstrap_bytes::<C>() >= query_call_slots() + native_query_callee_slots(),
            "lanes={lanes}"
        );
        assert!(q.control_bytes >= table_preparation_call_slots());
        assert!(q.control_bytes >= native_issuance_call_slots());
        assert!(q.control_bytes >= role_transition_call_slots());
        assert_eq!(q.slot_bytes, arc_inventory::<RetainedSlot>());
        // Five actual Box payloads, all retained frame/Step slots, and handoff
        // origins. This expected total never reads q.state_bytes as an input.
        let state = 5 * lanes * size_of::<u64>()
            + slots![
                [Frame<C>; 11],
                [Step<C>; 11],
                Option<RootValue<C>>,
                Role<C>,
                Output<C>,
                [consuming::LookupValueFor<C>; 2],
            ];
        assert_eq!(q.state_bytes, state);
        assert!(q.table_bytes >= arc_inventory::<RetainedTables>() + 512 * size_of::<u64>());
        // Cross-check the aggregate against concrete region-side obligations,
        // not sum(q.*). Native child keys/tables/scratch stay additional owned
        // contracts; this lower bound must not be called full child admission.
        let retained = arc_inventory::<RetainedTables>()
            + 512 * size_of::<u64>()
            + 2 * arc_inventory::<RetainedSlot>()
            + arc_inventory::<RetainedCoordination>()
            + 2 * state
            + slots![Mask<C>, Coordinator<C>, [Role<C>; 2]];
        let transition = table_preparation_call_slots()
            .max(native_issuance_call_slots())
            .max(role_transition_call_slots());
        assert!(q.total >= retained + transition);
        assert_eq!(
            q.incoming_owner_bytes,
            size_of::<Input>() + input.payload().capacity() * size_of::<i64>()
        );
        assert_eq!(
            q.work.input_retirement_words,
            lanes + input.payload().capacity()
        );
        assert_eq!(q.peer_bytes, 1688 + 208 * lanes);
        drop((original, input));
    }
}

fn check_original_storage(
    role: &Role<C>,
    root: &OwnedBudget,
    pointer: *const i64,
    lanes: usize,
    capacity: usize,
) {
    let state = role.slot.state.lock().unwrap();
    assert!(std::ptr::eq(state.lease.budget.budget(), root.budget()));
    assert!(std::ptr::eq(
        state.tables._lease.budget.budget(),
        root.budget()
    ));
    assert!(std::ptr::eq(state.tables.input.budget(), root.budget()));
    assert_eq!(state.tables.input.payload().as_ptr(), pointer);
    assert_eq!(state.tables.input.payload().capacity(), capacity);
    assert_eq!(state.tables.input.payload().len(), lanes);
    for work in &state.work {
        assert_eq!(work.len(), lanes); // Box payload: no spare capacity to omit.
    }
    let RootValue::Clear(original) = state.root.as_ref().unwrap() else {
        panic!("not original clear root")
    };
    assert!(std::ptr::eq(original._lease.budget.budget(), root.budget()));
    match &original.backing {
        CommonBackingFor::Root(values) => {
            assert_eq!(values.len(), lanes);
            assert_eq!(values.capacity(), lanes); // Actual retained native Vec.
        }
        _ => panic!("clear-root issuer did not retain original root Vec"),
    }
    assert_eq!(
        state.tables.checked.values(tables::Table::LowByte).len(),
        256
    );
    assert_eq!(
        state.tables.checked.values(tables::Table::HighByte).len(),
        256
    );
}

#[test]
fn genuine_clear_roots_retain_actual_capacity_counter_and_source_identity() {
    for lanes in [1, 2, 3, 127, 128, 129, 255, 256] {
        let root = OwnedBudget::new(64 << 20).unwrap();
        let initial = root.budget().used_bytes();
        let source = source();
        let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
        let values = vec![65535; lanes];
        let input = owner(&root, &values, 23);
        let pointer = input.payload().as_ptr();
        let capacity = input.payload().capacity();
        let incoming = input.charged_bytes();
        let s = spec(source, lanes);
        let (q, working, original) = owned_working(&dealer, &root, &s, &input);
        let admitted = root.budget().used_bytes();
        let mut maximum = admitted;
        let mut entropy = seeded();
        let (mask, coordinator, roles) = construct_with(&mut dealer, s, input, working, |bytes| {
            maximum = maximum.max(root.budget().used_bytes());
            entropy(bytes)
        })
        .unwrap();
        assert_eq!(q.incoming_owner_bytes, incoming);
        assert_eq!(maximum, admitted); // No counter growth at any entropy site.
        assert!(root.budget().used_bytes() <= admitted);
        for role in &roles {
            check_original_storage(role, &root, pointer, lanes, capacity);
        }
        {
            let left = roles[0].slot.state.lock().unwrap();
            let right = roles[1].slot.state.lock().unwrap();
            assert!(Arc::ptr_eq(&left.tables, &right.tables));
        }
        drop((mask, coordinator, roles, original, dealer));
        assert_eq!(root.budget().used_bytes(), initial);
    }
}

#[test]
fn cancellation_does_not_release_original_source_while_frame_custody_survives() {
    let root = OwnedBudget::new(64 << 20).unwrap();
    let initial = root.budget().used_bytes();
    let source = source();
    let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
    let input = owner(&root, &[0, 65535, (1i64 << 38) - 1], 257);
    let pointer = input.payload().as_ptr();
    let input_bytes = input.charged_bytes();
    let s = spec(source, 3);
    let (_, working, original) = owned_working(&dealer, &root, &s, &input);
    let (mask, coordinator, mut roles) =
        construct_with(&mut dealer, s, input, working, seeded()).unwrap();
    let weak = {
        let state = roles[0].slot.state.lock().unwrap();
        Arc::downgrade(&state.tables)
    };
    let frame = roles[0].begin(&mut || false).unwrap();
    drop((mask, coordinator, roles, original, dealer));
    assert!(frame.bytes().is_err()); // Revocation is not a storage-retirement credit.
    {
        let tables = weak.upgrade().expect("frame must retain original source");
        assert_eq!(tables.input.payload().as_ptr(), pointer);
        assert_eq!(tables.input.payload().capacity(), 260);
        assert!(std::ptr::eq(tables.input.budget(), root.budget()));
        assert_eq!(tables.input.charged_bytes(), input_bytes);
    }
    assert!(root.budget().used_bytes() >= initial + input_bytes);
    drop(frame);
    assert!(weak.upgrade().is_none());
    assert_eq!(root.budget().used_bytes(), initial);
}

#[test]
fn actual_eleven_round_frames_and_handoffs_stay_with_original_admitted_counter() {
    for lanes in [1, 3, 128, 256] {
        let root = OwnedBudget::new(64 << 20).unwrap();
        let initial = root.budget().used_bytes();
        let source = source();
        let mut dealer = OwnedLocalDealer::new(source, [4; 32], [92; 32], &root).unwrap();
        let input = owner(&root, &vec![65535; lanes], 7);
        let s = spec(source, lanes);
        let (q, working, original) = owned_working(&dealer, &root, &s, &input);
        let ceiling = root.budget().used_bytes();
        let (mask, coordinator, mut roles) =
            construct_with(&mut dealer, s, input, working, seeded()).unwrap();
        let mut pair = [
            roles[0].begin(&mut || false).unwrap(),
            roles[1].begin(&mut || false).unwrap(),
        ];
        // Harness custody only; no production Vec or unpriced metadata added.
        let mut retained: [Option<[Frame<C>; 2]>; 11] = std::array::from_fn(|_| None);
        let mut bytes = 0;
        for (round, slot) in retained.iter_mut().enumerate() {
            bytes += pair
                .iter()
                .map(|frame| frame.bytes().unwrap().len())
                .sum::<usize>();
            let left = roles[0]
                .advance(pair[1].bytes().unwrap(), &mut || false)
                .unwrap();
            let right = roles[1]
                .advance(pair[0].bytes().unwrap(), &mut || false)
                .unwrap();
            *slot = Some(pair);
            assert!(root.budget().used_bytes() <= ceiling);
            if round == 10 {
                assert!(matches!((left, right), (Step::Ready, Step::Ready)));
                break;
            }
            pair = match (left, right) {
                (Step::Frame(left), Step::Frame(right)) => [left, right],
                _ => panic!("premature nExp handoff"),
            };
        }
        assert_eq!(bytes, 1688 + 208 * lanes);
        assert_eq!(q.peer_bytes, bytes);
        let [left, right] = roles;
        let outputs = [left.finish().unwrap(), right.finish().unwrap()];
        for output in &outputs {
            let state = output.slot.state.lock().unwrap();
            assert_eq!(state.phase, 255);
            assert!(state.root.is_none());
            assert!(state.work.iter().all(|work| work.len() == lanes));
            assert!(state
                .work
                .iter()
                .flat_map(|work| work.iter())
                .all(|word| *word == 0));
            assert!(std::ptr::eq(state.tables.input.budget(), root.budget()));
            assert!(std::ptr::eq(
                state.output.as_ref().unwrap()._lease.budget.budget(),
                root.budget()
            ));
        }
        drop((outputs, mask, coordinator, retained, original, dealer));
        assert_eq!(root.budget().used_bytes(), initial);
    }
}
