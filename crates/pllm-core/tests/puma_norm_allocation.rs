//! Dedicated allocation-failure/erasure audit for norm admission's shared kernel.
use pllm_core::puma::multiply::{
    norm::RowEnergyProof, InputBounds, Interval, MatrixShape, Operand, PartyRuntime, SetupParty,
    Status,
};
use pllm_core::puma::prss::PrssLimits;
use pllm_core::puma::{Context, Limits, Party};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

#[derive(Clone, Copy, Default)]
struct Slot {
    address: usize,
    bytes: usize,
}
#[derive(Clone, Copy, Default)]
struct Audit {
    active: bool,
    collect: bool,
    fail: Option<usize>,
    slots: [Slot; 16],
    allocations: usize,
    frees: usize,
    bad_erasure: bool,
}
thread_local! { static AUDIT: Cell<Audit> = Cell::new(Audit::default()); }
struct Allocator;
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let fail = AUDIT
            .try_with(|state| {
                let mut a = state.get();
                let fail = a.collect && a.fail == Some(0);
                if a.collect {
                    a.fail = a.fail.and_then(|n| n.checked_sub(1));
                }
                state.set(a);
                fail
            })
            .unwrap_or(false);
        if fail {
            return std::ptr::null_mut();
        }
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let _ = AUDIT.try_with(|state| {
                let mut a = state.get();
                if a.collect {
                    if let Some(slot) = a.slots.iter_mut().find(|s| s.address == 0) {
                        *slot = Slot {
                            address: pointer as usize,
                            bytes: layout.size(),
                        };
                    } else {
                        a.bad_erasure = true;
                    }
                    a.allocations += 1;
                }
                state.set(a);
            });
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let _ = AUDIT.try_with(|state| {
            let mut a = state.get();
            if a.active {
                if let Some(slot) = a.slots.iter_mut().find(|s| s.address == pointer as usize) {
                    // Every allocation made by begin is a secret Words buffer.
                    a.bad_erasure |= unsafe { std::slice::from_raw_parts(pointer, slot.bytes) }
                        .iter()
                        .any(|&v| v != 0);
                    *slot = Slot::default();
                    a.frees += 1;
                }
            }
            state.set(a);
        });
        unsafe {
            System.dealloc(pointer, layout);
        }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn owners() -> [PartyRuntime; 3] {
    let context = Context::new([73; 32], [29; 32], 9).unwrap();
    let mut setups = [Party::P0, Party::P1, Party::P2]
        .map(|p| SetupParty::new(context, p, PrssLimits::default()).unwrap());
    let public = setups.each_mut().map(|p| p.contribution().unwrap());
    for (i, p) in setups.iter_mut().enumerate() {
        for (j, message) in public.iter().enumerate() {
            if i != j {
                p.receive_public(*message).unwrap();
            }
        }
    }
    let mut seeds = setups.each_mut().map(|p| Some(p.share_seed().unwrap()));
    for (i, p) in setups.iter_mut().enumerate() {
        p.receive_seed(seeds[(i + 1) % 3].take().unwrap()).unwrap();
    }
    setups.each_mut().map(|p| p.finish().unwrap())
}

#[test]
fn norm_admission_allocates_nothing_and_begin_failure_sweep_erases() {
    let b = InputBounds {
        left: Interval::new(-8, 8).unwrap(),
        right: Interval::new(-8, 8).unwrap(),
    };
    for rescale in [false, true] {
        for fail in 0..5 {
            let mut owners = owners();
            AUDIT.set(Audit {
                active: true,
                collect: true,
                ..Audit::default()
            });
            let proof = RowEnergyProof::new(3, b, 192, 192).unwrap();
            let shape = MatrixShape::new(2, 3, 2).unwrap();
            let plan = if rescale {
                owners[0].plan_norm_matmul(shape, proof)
            } else {
                owners[0].plan_ring_norm_matmul(shape, proof)
            }
            .unwrap();
            let admission = AUDIT.replace(Audit::default());
            assert_eq!(admission.allocations, 0);
            let left = plan
                .share_operand(
                    Operand::Left,
                    Party::P0,
                    &[17; 6],
                    &[23; 6],
                    Limits::default(),
                )
                .unwrap();
            let right = plan
                .share_operand(
                    Operand::Right,
                    Party::P0,
                    &[31; 6],
                    &[47; 6],
                    Limits::default(),
                )
                .unwrap();
            let usage = owners[0].payload_usage();
            AUDIT.set(Audit {
                active: true,
                collect: true,
                fail: Some(fail),
                ..Audit::default()
            });
            let result = owners[0].begin(left, right);
            AUDIT.with(|state| {
                let mut a = state.get();
                a.collect = false;
                state.set(a);
            });
            let succeeded = result.is_ok();
            if succeeded {
                assert_eq!(owners[0].status(), Status::AwaitingRotate);
            } else {
                assert_eq!(owners[0].status(), Status::Retired);
            }
            owners[0].cancel();
            // Exported frame remains charged after cancellation until erased.
            if succeeded {
                assert_eq!(usage.live_bytes(), plan.lanes() * 8);
            }
            drop(result);
            let audit = AUDIT.replace(Audit::default());
            assert!(!audit.bad_erasure);
            assert_eq!(audit.allocations, audit.frees);
            assert!(audit.slots.iter().all(|s| s.address == 0));
            assert_eq!(usage.live_bytes(), 0);
            assert_eq!(owners[0].cursors().pair_blocks, 2);
            assert_eq!(owners[0].cursors().aes_work, 4);
            assert_eq!(owners[0].cursors().public_blocks, 0);
        }
    }
}
