//! Private D43 interval scan over the ORIGINAL unsigned LESS implementation.
//! Raw arithmetic only: no input/range/source authority, correction, reservation,
//! one-use/replay enforcement or authenticated S>=1 assertion. Parent prices key
//! material and the existing LESS/AES workspace in ADDITION to these owners.
use super::{Error, GpuKey, Profile};
use std::mem::size_of;
use zeroize::{Zeroize, Zeroizing};

const N: u64 = 1 << 50;
const MASK: u64 = N - 1;
const EXP_MASK: u64 = (1 << 13) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScanWork {
    pub less_queries: usize,
    pub memberships: usize,
    pub count_updates: usize,
    pub weighted_updates: usize,
    pub aes_blocks: usize,
}
impl ScanWork {
    fn covers(self, required: Self) -> bool {
        self.less_queries >= required.less_queries
            && self.memberships >= required.memberships
            && self.count_updates >= required.count_updates
            && self.weighted_updates >= required.weighted_updates
            && self.aes_blocks >= required.aes_blocks
    }
}

/// No Clone, public constructor or key/lease extraction. Fields consumed by the
/// genuine enclosing private receipt; this result itself grants no authority.
pub(crate) struct ScanResult {
    pub count: Zeroizing<i64>,
    pub sums: Zeroizing<[u64; 2]>,
}

struct Scratch {
    first: u8,
    previous: u8,
    current: u8,
    membership: u8,
    queried_bit: u8,
    first_endpoint: u64,
    previous_endpoint: u64,
    current_endpoint: u64,
    query: u64,
    count: i64,
    sums: [u64; 2],
    exponent: u64,
    multiplier: u64,
}
impl Zeroize for Scratch {
    fn zeroize(&mut self) {
        self.first.zeroize();
        self.previous.zeroize();
        self.current.zeroize();
        self.membership.zeroize();
        self.queried_bit.zeroize();
        self.first_endpoint.zeroize();
        self.previous_endpoint.zeroize();
        self.current_endpoint.zeroize();
        self.query.zeroize();
        self.count.zeroize();
        self.sums.zeroize();
        self.exponent.zeroize();
        self.multiplier.zeroize();
        #[cfg(test)]
        ERASED_SCRATCH.with(|n| n.set(n.get() + 1));
    }
}
impl Scratch {
    fn empty() -> Self {
        Self {
            first: 0,
            previous: 0,
            current: 0,
            membership: 0,
            queried_bit: 0,
            first_endpoint: 0,
            previous_endpoint: 0,
            current_endpoint: 0,
            query: 0,
            count: 0,
            sums: [0; 2],
            exponent: 0,
            multiplier: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScanResources {
    pub scratch_bytes: usize,
    pub result_bytes: usize,
    pub scan_arguments_bytes: usize,
    pub prefix_arguments_bytes: usize,
    pub loop_control_bytes: usize,
    pub move_return_bytes: usize,
    pub resource_query_bytes: usize,
    /// Named logical owners only, NOT compiled stack, allocator, AES or RSS.
    pub known_peak_bytes: usize,
    pub heap_bytes: usize,
    pub work: ScanWork,
}
// Public controls live alongside scratch. Both role and masked opening are
// public, but arithmetic accumulators/prefix bits remain in Zeroizing<Scratch>.
struct LoopControl {
    boundaries: std::ops::Range<usize>,
    index: usize,
    end: u64,
    role: u8,
    sign: i64,
}

pub(crate) fn resources() -> Result<ScanResources, Error> {
    let work = ScanWork {
        less_queries: 51,
        memberships: 51,
        count_updates: 51,
        weighted_updates: 102,
        aes_blocks: 51usize
            .checked_mul(Profile::new(50)?.resources().point_or_less_aes_blocks)
            .ok_or(Error::Resource)?,
    };
    let scratch_bytes = size_of::<Zeroizing<Scratch>>();
    let result_bytes = size_of::<ScanResult>();
    let scan_arguments_bytes = size_of::<(&GpuKey, u64, ScanWork)>();
    let prefix_arguments_bytes = size_of::<(&GpuKey, u64, &mut Scratch)>();
    let loop_control_bytes = size_of::<LoopControl>();
    // Source-level construction/return positions: scratch initializer + wrapper
    // move; result initializer + Result + caller Result move; scalar LESS Result
    // and returned prefix positions. No hidden padding or guessed native frames.
    let move_return_bytes = [
        size_of::<Scratch>(),
        size_of::<Zeroizing<Scratch>>(),
        size_of::<ScanResult>(),
        size_of::<Result<ScanResult, Error>>(),
        size_of::<Result<ScanResult, Error>>(),
        size_of::<Result<u8, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<u8>(),
        size_of::<u8>(),
        size_of::<Zeroizing<i64>>(),
        size_of::<Zeroizing<[u64; 2]>>(),
        size_of::<&mut std::ops::Range<usize>>(),
        size_of::<Option<usize>>(),
    ]
    .into_iter()
    .try_fold(0usize, |n, b| n.checked_add(b).ok_or(Error::Resource))?;
    // resources()'s owned return and scan's retained local, checked-profile and
    // native-resource locals, work/ref arguments and resource-result positions.
    let resource_query_bytes = [
        size_of::<ScanResources>(),
        size_of::<ScanResources>(),
        size_of::<Result<ScanResources, Error>>(),
        size_of::<Profile>(),
        size_of::<Result<Profile, Error>>(),
        size_of::<super::Resources>(),
        size_of::<ScanWork>(),
        size_of::<ScanWork>(),
        size_of::<usize>(),
        size_of::<Result<usize, Error>>(),
        size_of::<std::array::IntoIter<usize, 13>>(),
    ]
    .into_iter()
    .try_fold(0usize, |n, b| n.checked_add(b).ok_or(Error::Resource))?;
    let known_peak_bytes = [
        scratch_bytes,
        result_bytes,
        scan_arguments_bytes,
        prefix_arguments_bytes,
        loop_control_bytes,
        move_return_bytes,
        resource_query_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |n, b| n.checked_add(b).ok_or(Error::Resource))?;
    Ok(ScanResources {
        scratch_bytes,
        result_bytes,
        scan_arguments_bytes,
        prefix_arguments_bytes,
        loop_control_bytes,
        move_return_bytes,
        resource_query_bytes,
        known_peak_bytes,
        heap_bytes: 0,
        work,
    })
}

fn prefix(key: &GpuKey, endpoint: u64, scratch: &mut Scratch) -> Result<u8, Error> {
    scratch.query = if endpoint == 0 { 0 } else { endpoint - 1 };
    // Even endpoint zero executes ORIGINAL less(0); query count never shortens.
    scratch.queried_bit = key.less(scratch.query)?;
    Ok(if endpoint == 0 {
        0
    } else {
        key.party ^ scratch.queried_bit
    })
}

pub(crate) fn scan(key: &GpuKey, masked: u64, allowance: ScanWork) -> Result<ScanResult, Error> {
    if key.profile.bits() != 50 || key.party > 1 || masked >= N {
        return Err(Error::Domain);
    }
    let required = resources()?;
    if !allowance.covers(required.work) {
        return Err(Error::Resource);
    }
    let mut scratch = Zeroizing::new(Scratch::empty());
    let mut control = LoopControl {
        boundaries: 0..51,
        index: 0,
        end: 0,
        role: key.party,
        sign: 1 - 2 * i64::from(key.party),
    };
    scratch.first_endpoint = 0u64.wrapping_sub(masked) & MASK;
    scratch.previous_endpoint = scratch.first_endpoint;
    scratch.first = prefix(key, scratch.first_endpoint, &mut scratch)?;
    scratch.previous = scratch.first;
    for index in control.boundaries.by_ref() {
        control.index = index;
        control.end = if index == 0 { 1 } else { 1u64 << index };
        scratch.current_endpoint = control.end.wrapping_sub(masked) & MASK;
        scratch.current = if index == 50 {
            scratch.first
        } else {
            prefix(key, scratch.current_endpoint, &mut scratch)?
        };
        scratch.membership = scratch.previous
            ^ scratch.current
            ^ if scratch.current_endpoint < scratch.previous_endpoint {
                control.role
            } else {
                0
            };
        scratch.count += control.sign * i64::from(scratch.membership);
        scratch.exponent = if index == 0 { 0 } else { (index - 1) as u64 };
        scratch.multiplier = if index == 0 { 0 } else { 1u64 << (50 - index) };
        scratch.sums[0] = scratch.sums[0].wrapping_add(
            (control.sign as u64)
                .wrapping_mul(u64::from(scratch.membership))
                .wrapping_mul(scratch.exponent),
        ) & EXP_MASK;
        scratch.sums[1] = scratch.sums[1].wrapping_add(
            (control.sign as u64)
                .wrapping_mul(u64::from(scratch.membership))
                .wrapping_mul(scratch.multiplier),
        ) & MASK;
        scratch.previous = scratch.current;
        scratch.previous_endpoint = scratch.current_endpoint;
    }
    Ok(ScanResult {
        count: Zeroizing::new(scratch.count),
        sums: Zeroizing::new(scratch.sums),
    })
}

#[cfg(test)]
std::thread_local! { static ERASED_SCRATCH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(bits: u8, point: u64, phase: u8) -> [GpuKey; 2] {
        GpuKey::issue_with(Profile::new(bits).unwrap(), point, |bytes| {
            bytes[..16].copy_from_slice(&0x123456789abcdef0123456789abcdef0u128.to_le_bytes());
            bytes[16..].copy_from_slice(
                &(0x65432100fedcba9865432100fedcba98u128 | u128::from(phase)).to_le_bytes(),
            );
            Ok(())
        })
        .unwrap()
    }
    fn payload(s: u64) -> [u64; 2] {
        if s == 0 {
            return [0, 0];
        }
        // Independent integer definition, no scan boundary iteration.
        let exponent = 63 - s.leading_zeros();
        [u64::from(exponent), (1u64 << 49) >> exponent]
    }
    #[test]
    fn fixed_pairs_cover_zero_powers_neighbors_top_wrap_and_both_root_phases() {
        for phase in 0..2 {
            for (point, masked) in [(0, 0), (1, 0), (N - 1, 2), (19, N - 20)] {
                let [left, right] = keys(50, point, phase);
                for plain in [
                    0,
                    1,
                    2,
                    3,
                    7,
                    8,
                    9,
                    (1 << 24) - 1,
                    1 << 24,
                    (1 << 24) + 1,
                    (1 << 49) - 1,
                    1 << 49,
                    N - 1,
                ] {
                    let opened = plain.wrapping_sub(point) & MASK;
                    let before = super::super::measured_aes_blocks();
                    let a = scan(&left, opened, resources().unwrap().work).unwrap();
                    let b = scan(&right, opened, resources().unwrap().work).unwrap();
                    assert_eq!(
                        super::super::measured_aes_blocks() - before,
                        2 * resources().unwrap().work.aes_blocks
                    );
                    let q = *a.count + *b.count;
                    assert!(q == 1 || q == -1);
                    let expected = payload(plain);
                    assert_eq!(
                        (a.sums[0] + b.sums[0]) & EXP_MASK,
                        (q as u64).wrapping_mul(expected[0]) & EXP_MASK
                    );
                    assert_eq!(
                        (a.sums[1] + b.sums[1]) & MASK,
                        (q as u64).wrapping_mul(expected[1]) & MASK
                    );
                }
                let a = scan(&left, masked, resources().unwrap().work).unwrap();
                let b = scan(&right, masked, resources().unwrap().work).unwrap();
                let q = *a.count + *b.count;
                let expected = payload(point.wrapping_add(masked) & MASK);
                assert!(q == 1 || q == -1);
                assert_eq!(
                    (a.sums[0] + b.sums[0]) & EXP_MASK,
                    (q as u64).wrapping_mul(expected[0]) & EXP_MASK
                );
                assert_eq!(
                    (a.sums[1] + b.sums[1]) & MASK,
                    (q as u64).wrapping_mul(expected[1]) & MASK
                );
            }
        }
    }
    #[test]
    fn every_one_short_mechanical_dimension_rejects_before_any_aes() {
        let pair = keys(50, 3, 0);
        let work = resources().unwrap().work;
        for field in 0..5 {
            let mut short = work;
            match field {
                0 => short.less_queries -= 1,
                1 => short.memberships -= 1,
                2 => short.count_updates -= 1,
                3 => short.weighted_updates -= 1,
                _ => short.aes_blocks -= 1,
            }
            let before = super::super::measured_aes_blocks();
            assert!(matches!(scan(&pair[0], 0, short), Err(Error::Resource)));
            assert_eq!(super::super::measured_aes_blocks(), before);
        }
        let before = super::super::measured_aes_blocks();
        scan(&pair[0], 0, work).unwrap();
        assert_eq!(
            super::super::measured_aes_blocks() - before,
            work.aes_blocks
        );
    }
    #[test]
    fn wrong_profile_role_and_masked_domain_refuse_before_query() {
        let mut pair = keys(50, 0, 1);
        let wrong = keys(13, 0, 1);
        let work = resources().unwrap().work;
        let before = super::super::measured_aes_blocks();
        assert!(matches!(scan(&wrong[0], 0, work), Err(Error::Domain)));
        assert!(matches!(scan(&pair[0], N, work), Err(Error::Domain)));
        pair[0].party = 2;
        assert!(matches!(scan(&pair[0], 0, work), Err(Error::Domain)));
        assert_eq!(super::super::measured_aes_blocks(), before);
    }
    #[test]
    fn result_erasure_and_scratch_drop_are_observable_without_key_extraction() {
        let pair = keys(50, 1, 0);
        let before = ERASED_SCRATCH.with(std::cell::Cell::get);
        let mut result = scan(&pair[0], N - 1, resources().unwrap().work).unwrap();
        assert_eq!(ERASED_SCRATCH.with(std::cell::Cell::get), before + 1);
        result.count.zeroize();
        result.sums.zeroize();
        assert_eq!(*result.count, 0);
        assert_eq!(*result.sums, [0, 0]);
        let mut scratch = Scratch::empty();
        scratch.count = 7;
        scratch.sums = [3, 9];
        scratch.first = 1;
        scratch.zeroize();
        assert_eq!((scratch.first, scratch.count, scratch.sums), (0, 0, [0, 0]));
    }
    #[test]
    fn native_default_expansion_and_codec_gates_remain_unchanged() {
        for bits in [8, 10] {
            assert!(Profile::new(bits)
                .unwrap()
                .resources()
                .full_expansion_bytes
                .is_some());
        }
        for bits in [13, 50] {
            assert_eq!(
                Profile::new(bits).unwrap().resources().full_expansion_bytes,
                None
            );
        }
        let pair = keys(50, 3, 0);
        let p = Profile::new(50).unwrap();
        let bytes = pair
            .into_iter()
            .next()
            .unwrap()
            .into_material(p.resources().codec_bytes)
            .unwrap();
        let restored = GpuKey::decode(p, 0, &bytes).unwrap();
        assert!(scan(&restored, 0, resources().unwrap().work).is_ok());
        assert!(Profile::new(50).unwrap().layernorm13_resources().is_err());
    }
    #[test]
    fn own_resources_are_concrete_logical_layouts_plus_original_less_work() {
        let r = resources().unwrap();
        assert_eq!(r.scratch_bytes, size_of::<Zeroizing<Scratch>>());
        assert_eq!(r.result_bytes, size_of::<ScanResult>());
        assert_eq!(
            r.scan_arguments_bytes,
            size_of::<(&GpuKey, u64, ScanWork)>()
        );
        assert_eq!(
            r.prefix_arguments_bytes,
            size_of::<(&GpuKey, u64, &mut Scratch)>()
        );
        assert_eq!(r.loop_control_bytes, size_of::<LoopControl>());
        assert_eq!(r.heap_bytes, 0);
        assert_eq!(
            (
                r.work.less_queries,
                r.work.memberships,
                r.work.count_updates,
                r.work.weighted_updates
            ),
            (51, 51, 51, 102)
        );
        assert_eq!(
            r.work.aes_blocks,
            51 * Profile::new(50)
                .unwrap()
                .resources()
                .point_or_less_aes_blocks
        );
    }
}
