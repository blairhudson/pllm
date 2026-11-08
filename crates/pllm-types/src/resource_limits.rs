//! Shared logical allocation policy, not an allocator/RSS or execution limit.
//! Extracted without changing the existing garbling public constant's value/type.
pub const MAX_ALLOCATION_BYTES: usize = 64 * 1024 * 1024;
