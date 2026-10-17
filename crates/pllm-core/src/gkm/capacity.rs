//! Admission policy, independent of cryptographic/numeric artifact identity.
use super::*;

/// Explicit resource profile. Existing constructors always select `Capsule`.
/// `Decoder` admits larger full-share geometry; it does not enable seeded windows.
/// Capacity is local policy, never a claim accepted from an untrusted wire frame.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Capacity {
    #[default]
    Capsule,
    Decoder,
}

// On 32-bit targets allocation/address-space limits can reject earlier. Avoid a
// truncating 4 GiB -> 0 conversion even for geometry-only admission there.
const FOUR_GIB: usize = if usize::BITS > 32 {
    (1u64 << 32) as usize
} else {
    usize::MAX
};

impl Capacity {
    pub const fn max_rows(self) -> usize {
        match self {
            Self::Capsule => MAX_DIMENSION,
            Self::Decoder => 262_144,
        }
    }
    pub const fn max_columns(self) -> usize {
        match self {
            Self::Capsule => MAX_DIMENSION,
            Self::Decoder => 16_384,
        }
    }
    pub const fn max_vocabulary(self) -> usize {
        match self {
            Self::Capsule => MAX_VOCABULARY,
            Self::Decoder => 262_144,
        }
    }
    pub const fn max_stages(self) -> usize {
        match self {
            Self::Capsule => MAX_STAGES,
            Self::Decoder => 256,
        }
    }
    pub const fn max_window_steps(self) -> usize {
        match self {
            Self::Capsule => MAX_WINDOW_STEPS,
            Self::Decoder => 65_536,
        }
    }
    pub const fn max_matrix_elements(self) -> usize {
        match self {
            Self::Capsule => MAX_DIMENSION * MAX_DIMENSION,
            Self::Decoder => 1 << 28,
        }
    }
    pub const fn max_source_bytes(self) -> usize {
        match self {
            Self::Capsule => MAX_PUBLICATION_BYTES,
            Self::Decoder => 2 << 30,
        }
    }
    pub const fn max_publication_bytes(self) -> usize {
        match self {
            Self::Capsule => MAX_PUBLICATION_BYTES,
            Self::Decoder => FOUR_GIB,
        }
    }
    pub const fn max_window_bytes(self) -> usize {
        match self {
            Self::Capsule => MAX_WINDOW_BYTES,
            Self::Decoder => 128 << 20,
        }
    }
    pub const fn max_process_publication_bytes(self) -> usize {
        match self {
            Self::Capsule => MAX_PROCESS_PUBLICATION_BYTES,
            Self::Decoder => FOUR_GIB,
        }
    }
    pub const fn max_process_wire_bytes(self) -> usize {
        match self {
            Self::Capsule => MAX_PROCESS_WIRE_BYTES,
            Self::Decoder => FOUR_GIB,
        }
    }
    pub const fn max_publication_wire_bytes(self) -> usize {
        match self {
            Self::Capsule => MAX_PUBLICATION_WIRE_BYTES,
            Self::Decoder => 32 << 20,
        }
    }
    pub const fn max_wire_frame_bytes(self) -> usize {
        match self {
            Self::Capsule => MAX_WIRE_FRAME_BYTES,
            Self::Decoder => 4 << 20,
        }
    }

    pub(super) fn matrix_elements(self, rows: usize, columns: usize) -> Result<usize, Error> {
        if rows == 0 || rows > self.max_rows() || columns == 0 || columns > self.max_columns() {
            return Err(Error::Limit);
        }
        rows.checked_mul(columns)
            .filter(|&n| n <= self.max_matrix_elements())
            .ok_or(Error::Limit)
    }

    pub(super) fn table_bytes(self, vocabulary: usize, width: usize) -> Result<usize, Error> {
        if vocabulary == 0
            || vocabulary > self.max_vocabulary()
            || width == 0
            || width > self.max_columns()
        {
            return Err(Error::Limit);
        }
        vocabulary
            .checked_mul(width)
            .and_then(|n| n.checked_mul(4))
            .filter(|&n| n <= self.max_source_bytes())
            .ok_or(Error::Limit)
    }
}
