//! NEW ordinary Cargo target; existing queued storage target stays unchanged.
//! Parent wiring: compile this target separately with --test staged_identity.
//! No HE key/context is instantiated by its tiny content/ABI regression fixtures.
#[path = "../src/bounded_allocator.rs"]
mod bounded_allocator;
#[path = "../src/staged_identity.rs"]
mod staged_identity;
#[path = "../src/staged_preparation.rs"]
mod staged_preparation;
#[path = "../src/staged_storage.rs"]
mod staged_storage;

#[global_allocator]
static ALLOCATOR: bounded_allocator::BoundedAllocator =
    bounded_allocator::BoundedAllocator::qualification();

// Type-check the actual WriterTo -> consuming content ticket -> ReaderFrom ->
// quarantined owner path. Never invoked: no HE-sized owner is allocated here,
// and successful byte identity deliberately cannot become preparation authority.
#[allow(dead_code)]
fn actual_owner_roundtrip(
    original: staged_preparation::RawRotation,
    destination: staged_preparation::RawRotation,
    output: &mut std::fs::File,
    input: std::fs::File,
) -> Result<(), staged_identity::IdentityError> {
    let mut export = staged_identity::RotationExportOwner::write_original(original, output)?;
    let ticket = export.preflight(input)?;
    let quarantine = ticket.inspect_existing(destination)?;
    let verdict = quarantine.into_preparation();
    export.release()?;
    verdict.map(|_| ())
}
