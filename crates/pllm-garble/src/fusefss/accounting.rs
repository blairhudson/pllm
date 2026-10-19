//! Test-only logical-capacity audit, not process RSS or a generic parser budget.
use super::*;
use serde_json::{json, Value as J};
use std::mem::size_of;

pub(super) fn gelu_owner_receipt() -> J {
    let d = Descriptor::gelu().unwrap();
    let dom = json::parse(d.canonical_bytes()).unwrap();
    let (dom_heap, largest_dom_block) = dom.logical_heap();
    let descriptor_heap = d.canonical.capacity()
        + d.boundaries.capacity() * size_of::<u64>()
        + d.pieces.capacity() * size_of::<Piece>()
        + d.expressions.capacity() * size_of::<Expr>();
    let source_bounds = 512 * size_of::<u64>();
    // fixture::Writer currently reserves MAX_DESCRIPTOR_BYTES. Validate observed
    // backing capacities, not just lengths, before making this exact-path claim.
    let mut writer_capacity = Vec::<u8>::new();
    reserve(&mut writer_capacity, MAX_DESCRIPTOR_BYTES).unwrap();
    let scratch_capacity = writer_capacity.capacity();
    let mut typed_bounds = Vec::<u64>::new();
    reserve(&mut typed_bounds, 512).unwrap();
    assert_eq!(typed_bounds.capacity() * size_of::<u64>(), source_bounds);
    let mut starts_capacity = Vec::<[u64; 2]>::new();
    reserve(&mut starts_capacity, d.shape().source_intervals).unwrap();
    assert_eq!(starts_capacity.capacity(), d.shape().source_intervals);
    // DOM growth reallocates one vector at a time, never shrinking; final DOM
    // bytes plus one final-largest block bounds overlap with the retired block.
    // Conservatively overlap *all* typed descriptor owners with that parser peak,
    // although actual parser growth completes before typed allocation starts.
    let constructor_upper = scratch_capacity
        + source_bounds
        + dom_heap
        + largest_dom_block
        + descriptor_heap
        + size_of::<Descriptor>()
        + size_of::<json::Value>()
        + 2 * size_of::<Vec<u8>>()
        + size_of::<Sha256>();
    let s = d.shape();
    let starts = s.source_intervals * size_of::<[u64; 2]>();
    let boundaries = (s.padded_intervals + 1) * size_of::<u64>();
    let ordinals = s.padded_intervals * size_of::<u64>();
    let coefficients = s.padded_intervals * size_of::<[u64; 2]>();
    // Query/VM vectors are empty for this exact descriptor. ordinals remain
    // compile scratch in production, not a retained DealerPlan allocation.
    assert_eq!((s.queries, s.vm_ops), (0, 0));
    let compile_heap = starts + boundaries + ordinals + coefficients;
    let compile_upper = descriptor_heap
        + compile_heap
        + size_of::<Descriptor>()
        + size_of::<DealerPlan>()
        + compile::emitter_owner_size()
        + 2 * size_of::<Vec<u64>>();
    let handoff_upper = descriptor_heap
        + boundaries
        + coefficients
        + size_of::<Descriptor>()
        + size_of::<DealerPlan>()
        + size_of::<DealerHandoff<'_>>();
    let upper = constructor_upper.max(compile_upper).max(handoff_upper);
    let reserved = 5 * (1 << 18); // Existing backend 1.25 MiB; not changed here.
    assert!(upper <= reserved);
    // Check actual production-shape vector capacities at every mask class. Test
    // plans additionally retain ordinals/mask, accounted separately below.
    for mask in [0, 1, 64, 1 << 49, (1 << 50) - 1] {
        let plan = d.compile(mask).unwrap();
        let actual = plan.test_owner_capacities();
        assert_eq!(actual, (boundaries, ordinals, coefficients, 0, 0));
    }
    json!({"schema":"pllm.fusefss.exact-gelu-logical-owner-audit.v1",
        "architecture":std::env::consts::ARCH,"descriptor_sha256":GELU_DIGEST_HEX,
        "type_layout_bytes":{"json_value":size_of::<json::Value>(),"piece":size_of::<Piece>(),
            "expression":size_of::<Expr>(),"descriptor":size_of::<Descriptor>(),
            "dealer_plan_test_build_conservative_for_production":size_of::<DealerPlan>(),
            "emitter":compile::emitter_owner_size(),"dealer_handoff":size_of::<DealerHandoff<'_>>()},
        "backend_existing_reservation_bytes":reserved,
        "public_descriptor":{"canonical_bytes":d.canonical.len(),"logical_heap_bytes":descriptor_heap,
            "canonical_capacity":d.canonical.capacity(),"boundary_capacity":d.boundaries.capacity(),
            "piece_capacity":d.pieces.capacity(),"expression_capacity":d.expressions.capacity()},
        "constructor":{"api":"Descriptor::gelu()","canonical_scratch_capacity_bytes":scratch_capacity,
            "source_boundary_scratch_bytes":source_bounds,"json_dom_heap_bytes":dom_heap,
            "json_dom_largest_allocation_bytes":largest_dom_block,"conservative_overlap_upper_bytes":constructor_upper},
        "identity_first_input_api":{"api":"Descriptor::from_frozen_gelu_bytes(bytes)",
            "rejects_alternates_before_dom":true,"input_buffer_charged_separately":true,
            "upper_including_exact_32900_byte_input":constructor_upper-scratch_capacity-source_bounds+GELU_CANONICAL_BYTES},
        "compile":{"starts_scratch_bytes":starts,"ordinal_scratch_bytes":ordinals,
            "retained_boundaries_bytes":boundaries,"retained_coefficients_bytes":coefficients,
            "predicate_and_vm_bytes":0,"conservative_overlap_upper_bytes":compile_upper},
        "with_dealer_handoff":{"compiler_allocated_bytes":0,"overlap_upper_bytes":handoff_upper,
            "backend_callback_copies":"not included; charge backend-owned keys/copies separately"},
        "peak_logical_owner_upper_bytes":upper,"headroom_within_existing_reservation_bytes":reserved-upper,
        "conditions":["one frozen descriptor construction/compilation at a time; no retained older compiler plans",
            "64-bit target and the observed capacities/type layouts asserted by this audit",
            "sum of vector-capacity layout bytes and named inline owner objects; not allocator metadata, RSS, or thread stack",
            "constructor scratch is retired before backend handoff; backend callback allocations charged separately",
            "generic from_canonical_bytes parsing is not admitted by this exact-GELU receipt"],
        "generic_parser_limitation":"fallible bounded byte/depth/node parsing, but no general simultaneous logical-owner reservation; caller input buffer and concurrent old descriptors/plans need separate accounting"})
}

#[test]
fn exact_frozen_gelu_logical_owner_allowance() {
    let receipt = gelu_owner_receipt();
    assert!(receipt["peak_logical_owner_upper_bytes"].as_u64().unwrap() <= 5 * (1 << 18));
}
