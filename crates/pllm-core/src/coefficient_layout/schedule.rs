//! Allocation-free public control geometry for the shared clear trace/merge.
//! No coefficient values, ciphertext capability, keys, or storage ownership.
use super::{LayoutError, MAX_INTERLEAVE_INPUTS};

/// One operation on caller-owned polynomials. Lane indices address N-word
/// scratch polynomials; the temporary polynomial is separate from all lanes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterleaveStep {
    /// Zero every padded lane, then scale actual inputs by inverse(stride).
    Normalize {
        count: usize,
        padded: usize,
        divisor: usize,
    },
    /// Copy lane to temporary, then add its automorphism back to that lane.
    Trace { lane: usize, element: usize },
    /// Read both lanes before overwriting left: temporary = left - X^shift*right,
    /// left += X^shift*right, then left += automorphism(temporary, element).
    Merge {
        left: usize,
        right: usize,
        shift: usize,
        element: usize,
    },
    /// Copy lane zero to output only after every trace and merge is complete.
    Finish,
}

/// Checked geometry, independent of modulus and coefficient contents. The
/// caller's degree gate remains explicit; the shared clear ceiling stays 8192.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterleaveSchedule {
    degree: usize,
    stride: usize,
    count: usize,
    padded: usize,
}

impl InterleaveSchedule {
    pub fn bounded(
        degree: usize,
        stride: usize,
        count: usize,
        max_degree: usize,
    ) -> Result<Self, LayoutError> {
        if !degree.is_power_of_two() || degree > max_degree || degree > 8192 {
            return Err(LayoutError::InvalidDegree);
        }
        if !stride.is_power_of_two()
            || stride > degree
            || count == 0
            || count > stride.min(MAX_INTERLEAVE_INPUTS)
        {
            return Err(LayoutError::InvalidGeometry);
        }
        Ok(Self {
            degree,
            stride,
            count,
            padded: count.next_power_of_two(),
        })
    }

    pub fn padded(self) -> usize {
        self.padded
    }

    /// Same padded lane storage plus one N-word temporary as the legacy kernel.
    pub fn scratch_words(self) -> usize {
        (self.padded + 1) * self.degree
    }

    pub fn steps(self) -> InterleaveSteps {
        InterleaveSteps {
            schedule: self,
            phase: 0,
            lane: 0,
            h: self.padded,
        }
    }
}

/// Constant-sized cursor. Trace order is lane-major, including structural-zero
/// lanes; merge levels descend, with ascending left lanes within each level.
#[derive(Clone, Copy, Debug)]
pub struct InterleaveSteps {
    schedule: InterleaveSchedule,
    phase: u8,
    lane: usize,
    h: usize,
}

impl Iterator for InterleaveSteps {
    type Item = InterleaveStep;

    fn next(&mut self) -> Option<Self::Item> {
        let s = self.schedule;
        if self.phase == 0 {
            self.phase = 1;
            return Some(InterleaveStep::Normalize {
                count: s.count,
                padded: s.padded,
                divisor: s.stride,
            });
        }
        if self.phase == 1 {
            if self.h < s.stride {
                let step = InterleaveStep::Trace {
                    lane: self.lane,
                    element: s.degree / self.h + 1,
                };
                self.h *= 2;
                if self.h == s.stride {
                    self.lane += 1;
                    self.h = s.padded;
                    if self.lane == s.padded {
                        self.phase = 2;
                        self.lane = 0;
                        self.h = s.padded / 2;
                    }
                }
                return Some(step);
            }
            self.phase = 2;
            self.lane = 0;
            self.h = s.padded / 2;
        }
        if self.phase == 2 {
            if self.h != 0 {
                let step = InterleaveStep::Merge {
                    left: self.lane,
                    right: self.lane + self.h,
                    shift: self.h,
                    element: s.degree / self.h + 1,
                };
                self.lane += 1;
                if self.lane == self.h {
                    self.lane = 0;
                    self.h /= 2;
                }
                return Some(step);
            }
            self.phase = 3;
            return Some(InterleaveStep::Finish);
        }
        None
    }
}

impl std::iter::FusedIterator for InterleaveSteps {}
