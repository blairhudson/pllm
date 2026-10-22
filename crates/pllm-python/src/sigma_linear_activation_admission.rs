//! Preconstruction audit for the private, unregistered same-process SIGMA core.
//!
//! **Blocked: no admitted input owner or factory is implemented here.** Existing
//! compiler execution admission starts after plan/inventory/request construction.
//! Its reservation cannot retroactively cover those constructors. This module is
//! mounted only for metadata-only regressions, never called by the bridge core.
//! These tests do not supply the missing inventory construction admission.
//!
//! # Actual owners and available queries
//!
//! * `DecoderPlan::checked_owned_backing_bytes` is an allocation-free census of
//!   an EXISTING plan: both graphs, operation/state/transform vectors, string and
//!   digest capacities, and the shared JSON-tree accounting policy. Add the root
//!   `size_of::<DecoderPlan>()` once. This is useful when moving that exact plan,
//!   not a quote for Clone, deserialization or model lowering. Model expansion
//!   limits and bounded canonical hashing do not admit those other operations.
//! * `DecoderMatrixInventory::retained_payload_bytes` includes the EXISTING
//!   private bytes Vec capacity, digest backing and inline owner. Its canonical
//!   byte slice exposes length, not that capacity. Both public constructors,
//!   `lower_decoder_matrix_inventory` and `decoder_matrix_inventory_bytes`, lack
//!   a preconstruction resource query and a reservation-bearing construction
//!   boundary. Lowering owns preflight/grouping maps and sets, source/group/member
//!   metadata and copies, allocating plan/config digests, canonical serialization
//!   scratch/output and the final digest. The byte entry point additionally owns
//!   duplicate-key sets/strings, serde decoding scratch and a temporary plan.
//! * `ExpectedSigmaSourcePin::checked_construction_quote` and
//!   `from_trusted_publisher_admitted` ARE public associated functions. Their
//!   `SigmaSourceConstructionResources` return type is re-exported by the
//!   compiler facade so enclosing owners can compose its concrete inventory.
//!   This quote covers the exact incoming artifact/selection/digest capacities,
//!   three exact string and two coordinate tuple copies, retained pin and named
//!   constructor/decoder scratch. Its numeric cap is explicitly NOT a reservation
//!   token. A containing owner must retain the original charge through pin drop.
//! * Source import/certification, owned compiler preflight and role-split quotes
//!   cover their subsequent phases only. Borrowed plan/inventory, source bytes,
//!   offsets, original f64 values, canonical i64 coefficients, input i64 values
//!   and operation-name owners remain outside that compiler admission. Request
//!   selection is a separate owned capacity, not the expected-pin selection.
//! * The existing bridge's private `BridgeResources` covers its concrete Rust
//!   Rc retention, coordinator/two roles, bounded frames/outputs and transactional
//!   packaging overlap. It excludes CPython objects and input extraction. It
//!   cannot serve as an input construction quote or a Python packaging allowance.
//!
//! # Exact stopping seam
//!
//! Starting with an already-owned exact plan does not require cloning it, but
//! constructing its genuine inventory still has no owner-provided peak quote
//! BEFORE lowering. A post-lowering retained census cannot price that earlier
//! overlap. Starting with BOTH an already-built plan and inventory instead only
//! recovers the existing Rust baseline; it does not solve preconstruction.
//!
//! Required next seam belongs to the inventory owner: an allocation-free checked
//! peak census over the same immutable plan, including actual resident capacities,
//! grouping/digest/serialization/error scratch, plus construction that enforces
//! that bound while the caller's original reservation remains live. Any plan
//! clone/JSON route needs its own corresponding owner contract. No fixed byte
//! allowance, doubled retained size, serialized byte cap or Arc<Budget> can replace
//! either contract. Therefore no aggregate number, success token, placeholder
//! factory or alternative paper-specific accounting framework is exposed here.
//!
//! Registration additionally needs actual immutable Python buffer custody during
//! decoding, constructor/clone scratch and Python packaging overlap composed with
//! original Budget guards, and reviewed integration with existing fail-closed
//! role transaction guards. No raw native escape, lifetime cast, D27 change or
//! SDK promotion follows from the source metadata constructor becoming available.

#[cfg(test)]
#[path = "sigma_linear_activation_admission_tests.rs"]
mod tests;
