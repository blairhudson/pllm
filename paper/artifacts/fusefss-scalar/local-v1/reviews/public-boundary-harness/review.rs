//! Bounded public-API checks. Expected results use original unfused equations.
//! No compiler/backend evaluation output defines an expected numeric value.
use pllm_garble::fusefss::Descriptor;
use pllm_garble::fusefss_backend::{Budget, CompiledGelu, Error, LookupBackend, MAX_ALLOCATION_BYTES};
use pllm_garble::shared_rescale_reference::Context;
use sha2::{Digest, Sha256};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering::SeqCst};

struct Meter;
static ENABLED: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static FAIL: AtomicIsize = AtomicIsize::new(-1);
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let fail = FAIL.load(SeqCst);
        if fail == 0 { return std::ptr::null_mut(); }
        if fail > 0 { FAIL.fetch_sub(1, SeqCst); }
        let pointer = System.alloc(layout);
        if !pointer.is_null() && ENABLED.load(SeqCst) {
            let next = LIVE.fetch_add(layout.size() as isize, SeqCst) + layout.size() as isize;
            assert!(next >= 0);
            PEAK.fetch_max(next as usize, SeqCst);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if ENABLED.load(SeqCst) { LIVE.fetch_sub(layout.size() as isize, SeqCst); }
        System.dealloc(pointer, layout);
    }
}
#[global_allocator]
static ALLOCATOR: Meter = Meter;
const N: u64 = 1 << 50;
const MASK: u64 = N - 1;

fn context(index: u64) -> Context {
    Context { plan: [11; 32], session: [22; 32], operation: [33; 32], first_tensor_index: index }
}
fn table() -> [u64; 256] {
    let record: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../paper/reproduction-contracts/fusefss/gelu-table.json")).unwrap();
    let values: [u64; 256] = std::array::from_fn(|i| record["values"][i].as_u64().unwrap());
    let mut hash = Sha256::new();
    for value in values { hash.update(value.to_le_bytes()); }
    assert_eq!(format!("{:x}", hash.finalize()),
        "b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b");
    values
}
fn oracle(word: u64, values: &[u64; 256]) -> u64 {
    let x = if word < N / 2 { word as i64 } else { word as i64 - N as i64 };
    let q = x.div_euclid(64);
    let residual = values[q.unsigned_abs().min(255) as usize];
    (if x >= 0 { word } else { 0 }).wrapping_sub(residual) & MASK
}

fn compiler_public_padding(values: &[u64; 256]) {
    let descriptor = Descriptor::gelu().unwrap();
    let public = descriptor.compile(0).unwrap();
    public.with_dealer_handoff(|h| {
        assert_eq!(h.shifted_boundaries.len(), 513);
        assert_eq!(&h.shifted_boundaries[..3], &[0, 1, 64]);
        assert_eq!(h.coefficients[0], h.coefficients[1]);
        assert_eq!(h.coefficients[0], [0, 1]);
        let mut independently_solved = vec![0, 1];
        independently_solved.extend((1..=255).map(|j| 64 * j));
        independently_solved.push(N / 2);
        independently_solved.extend((1..=254).rev().map(|j| N - 64 * j));
        independently_solved.push(N);
        assert_eq!(h.shifted_boundaries, independently_solved);
        for (i, interval) in h.shifted_boundaries.windows(2).enumerate() {
            let [c, a] = h.coefficients[i];
            for word in [interval[0], interval[1] - 1] {
                assert_eq!(a.wrapping_mul(word).wrapping_add(c) & MASK, oracle(word, values));
            }
        }
        Ok(())
    }).unwrap();
    println!("public compiler geometry: 512 intervals; duplicate [0,1)/[1,64); 1024 independent endpoint checks passed");
}

fn execute(gate: &CompiledGelu<'_>, words: &[u64], index: u64) -> Vec<u64> {
    let [mut a, mut b] = gate.issue(words.len(), context(index)).unwrap();
    let left = vec![MASK; words.len()];
    let right: Vec<_> = words.iter().map(|x| (x + 1) & MASK).collect();
    let af = a.start(&left).unwrap();
    let bf = b.start(&right).unwrap();
    let ap = a.advance(bf.as_bytes()).unwrap();
    let bp = b.advance(af.as_bytes()).unwrap();
    let ao = a.finish(bp.as_bytes()).unwrap();
    let bo = b.finish(ap.as_bytes()).unwrap();
    assert!(a.is_consumed() && b.is_consumed());
    assert!(matches!(a.start(&left), Err(Error::Consumed)));
    ao.shares().iter().zip(bo.shares()).map(|(x, y)| x.wrapping_add(*y) & MASK).collect()
}

fn protected_direct_oracle(values: &[u64; 256]) {
    let words = [0, 1, 2, 63, 64, 65, 4095, 4096, 4097, 16255, 16256, 16257, 16319, 16320,
        N/2-1, N/2, N/2+1, N-16321, N-16320, N-16257, N-16256, N-16255, N-4097, N-4096,
        N-65, N-64, N-63, N-2, N-1, (1<<37)-1, 1<<37, (1<<37)+64,
        N-(1<<37)-1, N-(1<<37), N-(1<<37)+1, 127, 128, 129, 8191, 8192];
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for backend in [LookupBackend::PublicBoundary, LookupBackend::PrivateVector] {
        let gate = CompiledGelu::with_backend(&budget, backend).unwrap();
        let resident = budget.used_bytes();
        let selected = if backend == LookupBackend::PublicBoundary { &words[..] } else { &words[..8] };
        for (chunk, input) in selected.chunks(4).enumerate() {
            let actual = execute(&gate, input, chunk as u64 * 4);
            for (x, y) in input.iter().zip(actual) { assert_eq!(y, oracle(*x, values)); }
            assert_eq!(budget.used_bytes(), resident);
        }
        println!("{backend:?}: {} direct-oracle protected outputs passed; fresh issuance per chunk/lane", selected.len());
    }
    assert_eq!(budget.used_bytes(), 0);
}

fn heap_and_leases(values: &[u64; 256], lanes: usize) {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    LIVE.store(0, SeqCst); PEAK.store(0, SeqCst); ENABLED.store(true, SeqCst);
    let gate = CompiledGelu::with_backend(&budget, LookupBackend::PublicBoundary).unwrap();
    let constructor_peak = PEAK.load(SeqCst);
    let resident_heap = LIVE.load(SeqCst);
    PEAK.store(resident_heap as usize, SeqCst); // phase peak, excluding constructor DOM
    let cost = gate.resources(lanes).unwrap();
    let descriptor_reservation = budget.used_bytes();
    let [mut a, mut b] = gate.issue(lanes, context(0)).unwrap();
    let issued_heap = LIVE.load(SeqCst);
    let issued_reserved = budget.used_bytes();
    let x = [0, 64, N-1, (1<<37)+64];
    let left = [MASK; 4];
    let right = [1, 65, 0, (1<<37)+65];
    let af = a.start(&left[..lanes]).unwrap();
    let bf = b.start(&right[..lanes]).unwrap();
    let masked_heap = LIVE.load(SeqCst);
    let ap = a.advance(bf.as_bytes()).unwrap();
    let bp = b.advance(af.as_bytes()).unwrap();
    let product_heap = LIVE.load(SeqCst);
    let ao = a.finish(bp.as_bytes()).unwrap();
    let bo = b.finish(ap.as_bytes()).unwrap();
    for (i, (u, v)) in ao.shares().iter().zip(bo.shares()).enumerate() {
        assert_eq!(u.wrapping_add(*v) & MASK, oracle(x[i], values));
    }
    let finish_heap = LIVE.load(SeqCst);
    drop((gate, a, b));
    let exported_heap = LIVE.load(SeqCst);
    let exported_reserved = budget.used_bytes();
    let expected_exported = 2 * (cost.issuance.masked_frame_bytes + cost.issuance.product_frame_bytes + 8 * lanes);
    assert_eq!(exported_heap as usize, expected_exported);
    assert_eq!(exported_reserved, expected_exported);
    drop((af, bf, ap, bp));
    assert_eq!(budget.used_bytes(), 16 * lanes);
    assert_eq!(LIVE.load(SeqCst), (16 * lanes) as isize);
    drop((ao, bo));
    ENABLED.store(false, SeqCst);
    assert_eq!(LIVE.load(SeqCst), 0); assert_eq!(budget.used_bytes(), 0);
    assert!(PEAK.load(SeqCst).max(constructor_peak) <= cost.combined_reservation_bytes);
    assert_eq!(cost.issuance.public_source_table_bytes, 12296);
    assert_eq!(cost.issuance.public_table_bytes_per_party, lanes * 12296);
    assert_eq!(cost.issuance.party_key_logical_bytes, lanes * 1372);
    println!("heap lanes={lanes}: constructor_peak={constructor_peak}, resident_heap={resident_heap}, descriptor_reserved={descriptor_reservation}, issued_heap={issued_heap}, issued_reserved={issued_reserved}, masked_heap={masked_heap}, product_heap={product_heap}, finish_heap={finish_heap}, exported_heap_and_reserved={exported_heap}, issuance_online_peak={}, combined_reserved={}, public_source={}, public_party_copies_each={}, private_key_payload_each={}", PEAK.load(SeqCst), cost.combined_reservation_bytes, cost.issuance.public_source_table_bytes, cost.issuance.public_table_bytes_per_party, cost.issuance.party_key_logical_bytes);
}

fn cross_backend_and_malformed() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let private = CompiledGelu::new(&budget).unwrap();
    let public = CompiledGelu::with_backend(&budget, LookupBackend::PublicBoundary).unwrap();
    for (own, other) in [(&public, &private), (&private, &public)] {
        let [mut a, _b] = own.issue(1, context(99)).unwrap();
        let [_c, mut d] = other.issue(1, context(99)).unwrap();
        let af = a.start(&[0]).unwrap();
        let df = d.start(&[64]).unwrap();
        assert_ne!(&af.as_bytes()[8..40], &df.as_bytes()[8..40]);
        let mut raw = df.as_bytes().to_vec();
        raw[40..72].copy_from_slice(&af.as_bytes()[40..72]); // control nonce; test binding itself
        assert!(matches!(a.advance(&raw), Err(Error::Context)));
        assert!(a.is_consumed());
        assert!(matches!(a.advance(df.as_bytes()), Err(Error::Consumed)));
    }
    for mutation in 0..6 {
        let [mut a, mut b] = public.issue(1, context(100)).unwrap();
        let af = a.start(&[0]).unwrap();
        let bf = b.start(&[64]).unwrap();
        let mut raw = bf.as_bytes().to_vec();
        match mutation {
            0 => raw = af.as_bytes().to_vec(),
            1 => { raw.pop(); },
            2 => raw.push(0),
            3 => raw[8] ^= 1,
            4 => raw[73] ^= 1,
            _ => raw[87] = 255, // noncanonical ring50 word
        }
        assert!(matches!(a.advance(&raw), Err(Error::Context)));
        assert!(a.is_consumed());
    }
    println!("cross-backend binding: both directions reject after nonce control; 6 malformed/reflected frames consume owner");
}

fn allocation_failures() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let gate = CompiledGelu::with_backend(&budget, LookupBackend::PublicBoundary).unwrap();
    let resident = budget.used_bytes();
    for site in 0..8 {
        FAIL.store(site, SeqCst);
        let result = gate.issue(1, context(200 + site as u64));
        FAIL.store(-1, SeqCst);
        assert!(matches!(result, Err(Error::Allocation) | Err(Error::Resource)));
        assert_eq!(budget.used_bytes(), resident);
    }
    let mut count = 0;
    for (phase, sites) in [(0, 3), (1, 4), (2, 1)] {
        for site in 0..sites {
            let [mut a, mut b] = gate.issue(1, context(count)).unwrap();
            if phase == 0 {
                FAIL.store(site, SeqCst);
                let result = a.start(&[1]);
                FAIL.store(-1, SeqCst);
                assert!(matches!(result, Err(Error::Allocation)));
            } else {
                let af = a.start(&[1]).unwrap();
                let bf = b.start(&[2]).unwrap();
                if phase == 1 {
                    FAIL.store(site, SeqCst);
                    let result = a.advance(bf.as_bytes());
                    FAIL.store(-1, SeqCst);
                    assert!(matches!(result, Err(Error::Allocation)));
                } else {
                    let _ap = a.advance(bf.as_bytes()).unwrap();
                    let bp = b.advance(af.as_bytes()).unwrap();
                    FAIL.store(site, SeqCst);
                    let result = a.finish(bp.as_bytes());
                    FAIL.store(-1, SeqCst);
                    assert!(matches!(result, Err(Error::Allocation)));
                }
            }
            assert!(a.is_consumed());
            drop((a, b));
            assert_eq!(budget.used_bytes(), resident);
            count += 1;
        }
    }
    println!("8 real issuance allocator failures retire both partial public copies/key owners; {count} real online allocator failure sites consume public lookup/party and refund after owner drop");
}

fn tight_preflight() {
    let probe = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let gate = CompiledGelu::with_backend(&probe, LookupBackend::PublicBoundary).unwrap();
    let cost = gate.resources(4).unwrap();
    let budget = Budget::new(cost.combined_reservation_bytes - 1).unwrap();
    let tight = CompiledGelu::with_backend(&budget, LookupBackend::PublicBoundary).unwrap();
    assert!(matches!(tight.issue(4, context(0)), Err(Error::Resource)));
    assert_eq!(budget.used_bytes(), cost.descriptor_reservation_bytes);
    println!("combined reservation minus one rejects before issuance; descriptor lease retained separately");
}

fn main() {
    let values = table();
    compiler_public_padding(&values);
    protected_direct_oracle(&values);
    heap_and_leases(&values, 1);
    heap_and_leases(&values, 4);
    cross_backend_and_malformed();
    allocation_failures();
    tight_preflight();
    println!("7 bounded public-API review groups passed; no security/full-model certification");
}
