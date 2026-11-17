use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
enum Behavior { Success, Eintr, Unwind, SharedUnwind }

struct Incoming {
    behavior: Behavior,
    calls: Arc<AtomicUsize>,
}

struct FakeOwner {
    incoming: Option<Incoming>,
    closed: bool,
}

impl Owner for FakeOwner {
    type Incoming = Incoming;
    fn new() -> Self { Self { incoming: None, closed: false } }
    fn vacant(&self) -> bool { self.incoming.is_none() && !self.closed }
    fn adopt_prepared(&mut self, incoming: Incoming) { self.incoming = Some(incoming); }
    fn close(&mut self) -> Result<(), ()> {
        let incoming = self.incoming.as_ref().unwrap();
        incoming.calls.fetch_add(1, Ordering::SeqCst);
        match incoming.behavior {
            Behavior::Success => { self.closed = true; Ok(()) }
            // Models the Core EINTR outcome, with no synthetic descriptor.
            Behavior::Eintr => Err(()),
            Behavior::Unwind => panic!("injected retirement unwind"),
            Behavior::SharedUnwind => panic!("shared transition hook was not invoked"),
        }
    }
    fn shared_retirement(&self) {
        if matches!(self.incoming.as_ref().unwrap().behavior, Behavior::SharedUnwind) {
            panic!("injected shared retirement unwind");
        }
    }
}

type Table = Mutex<Vec<EngineSlot<FakeOwner>>>;

fn acquire(table: &Table, behavior: Behavior) -> (Lease<FakeOwner>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let incoming = Incoming { behavior, calls: Arc::clone(&calls) };
    let lease = Engine::acquire(table, 0, || Ok::<_, ()>(incoming)).unwrap();
    (lease, calls)
}

#[test]
fn slot_count() {
    let registry = ResidentFds::prepare(3).unwrap();
    let slots = registry.slots.lock().unwrap();
    assert_eq!(slots.len(), 3);
    for slot in slots.iter() {
        assert_eq!(slot.phase, Phase::Vacant);
        match slot.custody.as_ref().unwrap() {
            Custody::Shared(entry) => {
                assert!(entry.fd.is_none());
                assert_eq!(Arc::strong_count(entry), 1);
            }
            Custody::Unique(_) => panic!("prebirth custody must be shared"),
        }
    }
    drop(slots);
    assert_eq!(registry.actual_capacity().unwrap(), 3);
    let empty = ResidentFds::prepare(0).unwrap();
    assert_eq!(empty.actual_capacity(), Ok(0));
    assert_eq!(empty.slots.lock().unwrap().len(), 0);
    assert!(matches!(ResidentFds::prepare(usize::MAX), Err(RegistryError::Overflow)));
    #[cfg(all(target_os = "macos", target_pointer_width = "64"))]
    {
        assert_eq!(std::mem::size_of::<libc::pthread_mutex_t>(), 64);
        assert_eq!(std::mem::align_of::<libc::pthread_mutex_t>(), 8);
        for count in [0, 3] {
            let quote = ResidentFds::quote(count).unwrap();
            let expected_registry = std::mem::size_of::<ResidentFds>() + 64;
            let expected_slots = count * std::mem::size_of::<Slot>();
            let expected_entries = count
                * std::mem::size_of::<ArcInnerModel<Entry<UnixCloseOnceFd>>>();
            assert_eq!(quote.registry_bytes, expected_registry);
            assert_eq!(quote.slot_backing_bytes, expected_slots);
            assert_eq!(quote.entry_backing_bytes, expected_entries);
            assert_eq!(quote.total_bytes,
                expected_registry + expected_slots + expected_entries);
        }
        assert_eq!(ResidentFds::quote(usize::MAX), Err(RegistryError::Overflow));
    }
    #[cfg(not(all(target_os = "macos", target_pointer_width = "64")))]
    {
        for count in [0, 3, usize::MAX] {
            assert_eq!(ResidentFds::quote(count), Err(RegistryError::Unknown));
        }
    }
    let table = Engine::<FakeOwner>::prepare(0).unwrap();
    assert!(matches!(Engine::acquire(&table, 0, || -> Result<Incoming, ()> {
        panic!("bounds must precede factory")
    }), Err(AcquireError::Registry(RegistryError::Bounds))));
}

#[test]
fn wrong_table() {
    let first = Engine::<FakeOwner>::prepare(1).unwrap();
    let second = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&first, Behavior::Success);
    let retained = lease.clone();
    let second_root = {
        let slots = second.lock().unwrap();
        match slots[0].custody.as_ref().unwrap() {
            Custody::Shared(entry) => Arc::as_ptr(entry),
            _ => unreachable!(),
        }
    };
    assert_eq!(Engine::retire(&second, lease), Err(RegistryError::WrongTable));
    let slots = second.lock().unwrap();
    assert_eq!(slots[0].phase, Phase::Vacant);
    match slots[0].custody.as_ref().unwrap() {
        Custody::Shared(entry) => assert_eq!(Arc::as_ptr(entry), second_root),
        _ => unreachable!(),
    }
    drop(slots);
    assert_eq!(Arc::strong_count(&retained.entry), 2);
    assert_eq!(Engine::retire(&first, retained), Ok(()));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn acquisition_success() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&table, Behavior::Success);
    let has_owner = lease.entry.fd.is_some();
    let phase = table.lock().unwrap()[0].phase;
    let refusal = Engine::acquire(&table, 0, || -> Result<Incoming, ()> {
        panic!("nonvacant factory")
    });
    let retired = Engine::retire(&table, lease);
    assert!(has_owner);
    assert_eq!(phase, Phase::Open);
    assert!(matches!(refusal, Err(AcquireError::Registry(RegistryError::State))));
    assert_eq!(retired, Ok(()));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    #[cfg(target_os = "macos")]
    {
        let registry = ResidentFds::prepare(1).unwrap();
        let lease = registry.acquire_with(0, || tempfile::tempfile().map(OwnedFd::from)).unwrap();
        let descriptor_present = lease.descriptor().is_some();
        let retired = registry.retire(lease);
        assert_eq!(retired, Ok(()));
        assert!(descriptor_present);
    }
}

#[test]
fn factory_error() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let original = {
        let slots = table.lock().unwrap();
        match slots[0].custody.as_ref().unwrap() {
            Custody::Shared(entry) => Arc::as_ptr(entry),
            _ => unreachable!(),
        }
    };
    let result = Engine::acquire(&table, 0, || Err::<Incoming, _>("factory failed"));
    assert!(matches!(result, Err(AcquireError::Factory("factory failed"))));
    let slots = table.lock().unwrap();
    assert_eq!(slots[0].phase, Phase::AcquisitionUnknown);
    match slots[0].custody.as_ref().unwrap() {
        Custody::Shared(entry) => {
            assert_eq!(Arc::as_ptr(entry), original);
            assert!(entry.fd.is_none());
        }
        _ => unreachable!(),
    }
    drop(slots);
    assert!(matches!(Engine::acquire(&table, 0, || -> Result<Incoming, ()> {
        panic!("quarantined factory")
    }), Err(AcquireError::Registry(RegistryError::State))));
}

#[test]
fn factory_unwind() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let original = {
        let slots = table.lock().unwrap();
        match slots[0].custody.as_ref().unwrap() {
            Custody::Shared(entry) => Arc::as_ptr(entry),
            _ => unreachable!(),
        }
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = Engine::acquire(&table, 0, || -> Result<Incoming, ()> {
            panic!("injected factory unwind")
        });
    }));
    assert!(result.is_err());
    assert!(table.is_poisoned());
    let slots = table.lock().err().unwrap().into_inner();
    assert_eq!(slots[0].phase, Phase::AcquisitionUnknown);
    match slots[0].custody.as_ref().unwrap() {
        Custody::Shared(entry) => {
            assert_eq!(Arc::as_ptr(entry), original);
            assert!(entry.fd.is_none());
            assert_eq!(Arc::strong_count(entry), 1);
        }
        _ => unreachable!(),
    }
}

#[test]
fn escaped_lease_exact_restoration() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&table, Behavior::Success);
    let escaped = lease.clone();
    let original = Arc::as_ptr(&escaped.entry);
    assert_eq!(Engine::retire(&table, lease), Err(RegistryError::Busy));
    let slots = table.lock().unwrap();
    assert_eq!(slots[0].phase, Phase::Open);
    match slots[0].custody.as_ref().unwrap() {
        Custody::Shared(entry) => {
            assert_eq!(Arc::as_ptr(entry), original);
            assert!(Arc::ptr_eq(entry, &escaped.entry));
            assert_eq!(Arc::strong_count(entry), 2);
        }
        _ => unreachable!(),
    }
    drop(slots);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(Engine::retire(&table, escaped), Ok(()));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn close_success() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&table, Behavior::Success);
    assert_eq!(Engine::retire(&table, lease), Ok(()));
    let slots = table.lock().unwrap();
    assert_eq!(slots[0].phase, Phase::Closed);
    match slots[0].custody.as_ref().unwrap() {
        Custody::Unique(entry) => assert!(entry.fd.as_ref().unwrap().closed),
        _ => panic!("successful retirement must retain unique custody"),
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    #[cfg(target_os = "macos")]
    {
        let registry = ResidentFds::prepare(1).unwrap();
        let lease = registry.acquire_with(0, || tempfile::tempfile().map(OwnedFd::from)).unwrap();
        let retired = registry.retire(lease);
        assert_eq!(retired, Ok(()));
        let slots = registry.slots.lock().unwrap();
        assert_eq!(slots[0].phase, Phase::Closed);
        match slots[0].custody.as_ref().unwrap() {
            Custody::Unique(entry) => {
                let owner = entry.fd.as_ref().unwrap();
                assert_eq!(owner.state(), CloseState::Closed);
                assert!(owner.descriptor().is_none());
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn injected_eintr() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&table, Behavior::Eintr);
    assert_eq!(Engine::retire(&table, lease), Err(RegistryError::Unknown));
    let slots = table.lock().unwrap();
    assert_eq!(slots[0].phase, Phase::Unknown);
    assert!(matches!(slots[0].custody, Some(Custody::Unique(_))));
    drop(slots);
    assert!(matches!(Engine::acquire(&table, 0, || -> Result<Incoming, ()> {
        panic!("unknown retry")
    }), Err(AcquireError::Registry(RegistryError::State))));
    drop(table);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn retirement_unwind() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&table, Behavior::Unwind);
    assert!(catch_unwind(AssertUnwindSafe(|| Engine::retire(&table, lease))).is_err());
    assert!(table.is_poisoned());
    let slots = table.lock().err().unwrap().into_inner();
    assert_eq!(slots[0].phase, Phase::Unknown);
    assert!(matches!(slots[0].custody, Some(Custody::Unique(_))));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn poison_retention() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&table, Behavior::Success);
    let identity = Arc::as_ptr(&lease.entry);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _lock = table.lock().unwrap();
        panic!("injected unrelated lock poison");
    })).is_err());
    assert_eq!(Engine::retire(&table, lease), Err(RegistryError::Poisoned));
    assert!(matches!(Engine::acquire(&table, 0, || -> Result<Incoming, ()> {
        panic!("poisoned factory")
    }), Err(AcquireError::Registry(RegistryError::Poisoned))));
    let slots = table.lock().err().unwrap().into_inner();
    assert_eq!(slots[0].phase, Phase::Open);
    match slots[0].custody.as_ref().unwrap() {
        Custody::Shared(entry) => {
            assert_eq!(Arc::as_ptr(entry), identity);
            assert_eq!(Arc::strong_count(entry), 1);
            assert!(entry.fd.is_some());
        }
        _ => unreachable!(),
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

fn assert_caller_shared(
    table: &Table, caller: &Option<Lease<FakeOwner>>,
    identity: *const Entry<FakeOwner>, phase: Phase, strong: usize,
) {
    let lease = caller.as_ref().expect("caller retains lease");
    assert_eq!(lease.index, 0);
    assert_eq!(Arc::as_ptr(&lease.entry), identity);
    let slots = table.lock().unwrap_or_else(|poison| poison.into_inner());
    assert_eq!(slots[0].phase, phase);
    match slots[0].custody.as_ref().unwrap() {
        Custody::Shared(entry) => {
            assert!(Arc::ptr_eq(entry, &lease.entry));
            assert_eq!(Arc::as_ptr(entry), identity);
            assert_eq!(Arc::strong_count(entry), strong);
            assert!(entry.fd.is_some());
        }
        _ => panic!("caller requires shared custody"),
    }
}

#[cfg(target_os = "macos")]
#[test]
fn try_retire_real_tempfile_success() {
    let registry = ResidentFds::prepare(1).unwrap();
    let mut caller = Some(registry.acquire_with(
        0, || tempfile::tempfile().map(OwnedFd::from)).unwrap());
    let identity = Arc::as_ptr(&caller.as_ref().unwrap().inner.entry);
    assert!(caller.as_ref().unwrap().descriptor().is_some());
    {
        let slots = registry.slots.lock().unwrap();
        assert_eq!(slots[0].phase, Phase::Open);
        match slots[0].custody.as_ref().unwrap() {
            Custody::Shared(entry) => {
                assert_eq!(Arc::as_ptr(entry), identity);
                assert_eq!(Arc::strong_count(entry), 2);
            }
            _ => panic!("open shared custody"),
        }
    }
    let escaped = caller.as_ref().unwrap().clone();
    assert_eq!(registry.try_retire(&mut caller), Err(RegistryError::Busy));
    assert_eq!(Arc::as_ptr(&caller.as_ref().unwrap().inner.entry), identity);
    assert!(Arc::ptr_eq(&caller.as_ref().unwrap().inner.entry, &escaped.inner.entry));
    drop(escaped);
    assert_eq!(registry.try_retire(&mut caller), Ok(()));
    assert!(caller.is_none());
    assert_eq!(registry.try_retire(&mut caller), Err(RegistryError::State));
    let slots = registry.slots.lock().unwrap();
    assert_eq!(slots[0].phase, Phase::Closed);
    match slots[0].custody.as_ref().unwrap() {
        Custody::Unique(entry) => {
            let owner = entry.fd.as_ref().unwrap();
            assert_eq!(owner.state(), CloseState::Closed);
            assert!(owner.descriptor().is_none());
        }
        _ => panic!("closed unique custody"),
    }
}

#[test]
fn try_retire_zero_and_multiple_escapes() {
    for count in [0, 1, 3] {
        let table = Engine::<FakeOwner>::prepare(1).unwrap();
        let (lease, calls) = acquire(&table, Behavior::Success);
        let identity = Arc::as_ptr(&lease.entry);
        let mut escaped: Vec<_> = (0..count).map(|_| lease.clone()).collect();
        let mut caller = Some(lease);
        assert_caller_shared(&table, &caller, identity, Phase::Open, count + 2);
        while !escaped.is_empty() {
            for _ in 0..2 {
                assert_eq!(Engine::try_retire(&table, &mut caller), Err(RegistryError::Busy));
                assert_caller_shared(&table, &caller, identity, Phase::Open, escaped.len() + 2);
                assert_eq!(calls.load(Ordering::SeqCst), 0);
            }
            drop(escaped.pop());
        }
        assert_caller_shared(&table, &caller, identity, Phase::Open, 2);
        assert_eq!(Engine::try_retire(&table, &mut caller), Ok(()));
        assert!(caller.is_none());
        let slots = table.lock().unwrap();
        assert_eq!(slots[0].phase, Phase::Closed);
        assert!(matches!(slots[0].custody, Some(Custody::Unique(_))));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn try_retire_none_wrong_table_bounds() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let other = Engine::<FakeOwner>::prepare(1).unwrap();
    let empty = Engine::<FakeOwner>::prepare(0).unwrap();
    let mut none: Option<Lease<FakeOwner>> = None;
    assert_eq!(Engine::try_retire(&empty, &mut none), Err(RegistryError::State));
    let (lease, calls) = acquire(&table, Behavior::Success);
    let identity = Arc::as_ptr(&lease.entry);
    let mut caller = Some(lease);
    assert_eq!(Engine::try_retire(&other, &mut caller), Err(RegistryError::WrongTable));
    assert_caller_shared(&table, &caller, identity, Phase::Open, 2);
    assert_eq!(other.lock().unwrap()[0].phase, Phase::Vacant);
    assert_eq!(Engine::try_retire(&empty, &mut caller), Err(RegistryError::Bounds));
    assert_caller_shared(&table, &caller, identity, Phase::Open, 2);
    caller.as_mut().unwrap().index = 1;
    assert_eq!(Engine::try_retire(&table, &mut caller), Err(RegistryError::Bounds));
    assert_eq!(caller.as_ref().unwrap().index, 1);
    assert_eq!(Arc::as_ptr(&caller.as_ref().unwrap().entry), identity);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    caller.as_mut().unwrap().index = 0;
    assert_eq!(Engine::try_retire(&table, &mut caller), Ok(()));
    assert!(caller.is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn try_retire_poison_preserves_caller() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&table, Behavior::Success);
    let identity = Arc::as_ptr(&lease.entry);
    let mut caller = Some(lease);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _lock = table.lock().unwrap();
        panic!("injected unrelated lock poison");
    })).is_err());
    for _ in 0..2 {
        assert_eq!(Engine::try_retire(&table, &mut caller), Err(RegistryError::Poisoned));
        assert_caller_shared(&table, &caller, identity, Phase::Open, 2);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
    let mut none: Option<Lease<FakeOwner>> = None;
    assert_eq!(Engine::try_retire(&table, &mut none), Err(RegistryError::State));
    caller.as_mut().unwrap().index = usize::MAX;
    assert_eq!(Engine::try_retire(&table, &mut caller), Err(RegistryError::Poisoned));
    assert_eq!(caller.as_ref().unwrap().index, usize::MAX);
    assert_eq!(Arc::as_ptr(&caller.as_ref().unwrap().entry), identity);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn try_retire_quarantine_and_closed_no_retry() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&table, Behavior::Success);
    let identity = Arc::as_ptr(&lease.entry);
    let mut caller = Some(lease);
    // Exercise phase refusal with matching identity before any extraction.
    for phase in [Phase::Vacant, Phase::AcquisitionUnknown, Phase::RetiringUnknown,
                  Phase::Unknown, Phase::Closed] {
        table.lock().unwrap()[0].phase = phase;
        assert_eq!(Engine::try_retire(&table, &mut caller), Err(RegistryError::State));
        assert_caller_shared(&table, &caller, identity, phase, 2);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
    table.lock().unwrap()[0].phase = Phase::Open;
    assert_eq!(Engine::try_retire(&table, &mut caller), Ok(()));
    assert!(caller.is_none());
    for _ in 0..2 {
        assert_eq!(Engine::try_retire(&table, &mut caller), Err(RegistryError::State));
        let slots = table.lock().unwrap();
        assert_eq!(slots[0].phase, Phase::Closed);
        assert!(matches!(slots[0].custody, Some(Custody::Unique(_))));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn try_retire_eintr_once() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&table, Behavior::Eintr);
    let identity = Arc::as_ptr(&lease.entry);
    let mut caller = Some(lease);
    assert_caller_shared(&table, &caller, identity, Phase::Open, 2);
    assert_eq!(Engine::try_retire(&table, &mut caller), Err(RegistryError::Unknown));
    assert!(caller.is_none());
    for _ in 0..2 {
        assert_eq!(Engine::try_retire(&table, &mut caller), Err(RegistryError::State));
        let slots = table.lock().unwrap();
        assert_eq!(slots[0].phase, Phase::Unknown);
        assert!(matches!(slots[0].custody, Some(Custody::Unique(_))));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    assert!(matches!(Engine::acquire(&table, 0, || -> Result<Incoming, ()> {
        panic!("quarantined factory");
    }), Err(AcquireError::Registry(RegistryError::State))));
}

#[test]
fn try_retire_unwind_shared_restores_unique_quarantines() {
    let shared = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&shared, Behavior::SharedUnwind);
    let identity = Arc::as_ptr(&lease.entry);
    let mut caller = Some(lease);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _ = Engine::try_retire(&shared, &mut caller);
    })).is_err());
    assert!(shared.is_poisoned());
    assert_caller_shared(&shared, &caller, identity, Phase::Open, 2);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(Engine::try_retire(&shared, &mut caller), Err(RegistryError::Poisoned));
    assert_caller_shared(&shared, &caller, identity, Phase::Open, 2);

    let unique = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&unique, Behavior::Unwind);
    let mut caller = Some(lease);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _ = Engine::try_retire(&unique, &mut caller);
    })).is_err());
    assert!(caller.is_none());
    assert!(unique.is_poisoned());
    assert_eq!(Engine::try_retire(&unique, &mut caller), Err(RegistryError::State));
    let slots = unique.lock().err().unwrap().into_inner();
    assert_eq!(slots[0].phase, Phase::Unknown);
    assert!(matches!(slots[0].custody, Some(Custody::Unique(_))));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn try_retire_concurrent_last_release() {
    let table = Engine::<FakeOwner>::prepare(1).unwrap();
    let (lease, calls) = acquire(&table, Behavior::Success);
    let identity = Arc::as_ptr(&lease.entry);
    let escaped = lease.clone();
    let mut caller = Some(lease);
    let (release, wait) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        wait.recv().unwrap();
        drop(escaped);
    });
    assert_eq!(Engine::try_retire(&table, &mut caller), Err(RegistryError::Busy));
    assert_caller_shared(&table, &caller, identity, Phase::Open, 3);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    release.send(()).unwrap();
    worker.join().unwrap();
    assert_caller_shared(&table, &caller, identity, Phase::Open, 2);
    assert_eq!(Engine::try_retire(&table, &mut caller), Ok(()));
    assert!(caller.is_none());
    let slots = table.lock().unwrap();
    assert_eq!(slots[0].phase, Phase::Closed);
    assert!(matches!(slots[0].custody, Some(Custody::Unique(_))));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
