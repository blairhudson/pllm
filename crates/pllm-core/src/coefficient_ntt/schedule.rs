//! Allocation-free public-index traversal shared by both arithmetic domains.
//! Only validated power-of-two degrees <=8192 reach these private iterators.
//! No coefficient, callback, owner, arithmetic backend, or heap is captured.

pub(super) struct SwapPairs {
    n: usize,
    next: usize,
    reversed: usize,
}

pub(super) fn swaps(n: usize) -> SwapPairs {
    SwapPairs { n, next: 1, reversed: 0 }
}

impl Iterator for SwapPairs {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.n {
            let i = self.next;
            self.next += 1;
            let mut bit = self.n >> 1;
            while self.reversed & bit != 0 {
                self.reversed ^= bit;
                bit >>= 1;
            }
            self.reversed ^= bit;
            if i < self.reversed {
                return Some((i, self.reversed));
            }
        }
        None
    }
}

pub(super) struct Stages {
    n: usize,
    next: usize,
}

pub(super) fn stages(n: usize) -> Stages {
    Stages { n, next: 0 }
}

pub(super) struct Stage {
    pub(super) index: usize,
    n: usize,
    len: usize,
}

impl Iterator for Stages {
    type Item = Stage;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.n.trailing_zeros() as usize {
            return None;
        }
        let index = self.next;
        self.next += 1;
        Some(Stage { index, n: self.n, len: 1 << (index + 1) })
    }
}

impl Stage {
    pub(super) fn blocks(&self) -> std::iter::StepBy<std::ops::Range<usize>> {
        (0..self.n).step_by(self.len)
    }

    pub(super) fn butterflies(&self, start: usize) -> Butterflies {
        Butterflies { start, half: self.len / 2, next: 0 }
    }
}

pub(super) struct Butterflies {
    start: usize,
    half: usize,
    next: usize,
}

impl Iterator for Butterflies {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.half {
            return None;
        }
        let low = self.start + self.next;
        self.next += 1;
        Some((low, low + self.half))
    }
}
