//! Insecure, deterministic fixtures for fresh compound-runtime graphs only.
//!
//! **Never use these owners with production secrets.** Every key is public:
//! role `i` uses little-endian u128 seeds `i + 1` (public contribution),
//! `i + 101` (private split), and `(i + 1) * 11` (own pairwise key).
//! Repeated calls intentionally reuse these keys and setup bindings. They must
//! remain isolated test graphs, not separate sessions on a shared transport.
//!
//! Available only with the nondefault `insecure-test-support` feature. Enabling
//! it does not change `SetupParty::new` or its independent OS entropy draws.

use super::{multiply, Context, Error, Party, PrssLimits, SeedEntropy, SeedPurpose, SetupParty};
use zeroize::Zeroizing;

struct PublicTestSeeds(Party);

impl SeedEntropy for PublicTestSeeds {
    fn draw(&mut self, purpose: SeedPurpose) -> Result<Zeroizing<[u8; 16]>, Error> {
        let role = self.0 as u128;
        let value = match purpose {
            SeedPurpose::Public => role + 1,
            SeedPurpose::Private => role + 101,
            SeedPurpose::Pairwise => (role + 1) * 11,
        };
        Ok(Zeroizing::new(value.to_le_bytes()))
    }
}

pub(super) fn fixed_setup(
    context: Context,
    party: Party,
    limits: PrssLimits,
) -> Result<SetupParty, Error> {
    SetupParty::with_entropy(context, party, limits, &mut PublicTestSeeds(party))
}

/// Create three fresh native compound owners, in P0/P1/P2 order, with PUBLIC keys.
///
/// Runs the original contribution broadcast, directed seed exchange and finish
/// protocol, including the ordinary compound `bound_context` namespace. The
/// caller supplies only public context and unchanged resource limits, never keys.
/// Two calls with identical arguments start with identical PRSS seeds/bindings
/// and zero cursors; identical frames/shares additionally require identical
/// inputs, operation schedules and routing order. No live state is copied/reset.
///
/// # Security
/// This fixture provides no secrecy. Keep both graphs isolated from production
/// and from each other's messages; identical bindings do not reject cross-talk.
pub fn fresh_compound_parties(
    context: Context,
    limits: PrssLimits,
) -> Result<[multiply::PartyRuntime; 3], Error> {
    let [p0, p1, p2] = [Party::P0, Party::P1, Party::P2]
        .map(|party| multiply::SetupParty::insecure_test_setup(context, party, limits));
    let mut setups = [p0?, p1?, p2?];
    let [p0, p1, p2] = setups.each_mut().map(|setup| setup.contribution());
    let contributions = [p0?, p1?, p2?];
    for (i, setup) in setups.iter_mut().enumerate() {
        for (j, contribution) in contributions.iter().enumerate() {
            if i != j {
                setup.receive_public(*contribution)?;
            }
        }
    }
    let [p0, p1, p2] = setups.each_mut().map(|setup| setup.share_seed());
    let mut seeds = [Some(p0?), Some(p1?), Some(p2?)];
    for (i, setup) in setups.iter_mut().enumerate() {
        setup.receive_seed(seeds[(i + 1) % 3].take().ok_or(Error::Phase)?)?;
    }
    let [p0, p1, p2] = setups.each_mut().map(|setup| setup.finish());
    Ok([p0?, p1?, p2?])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::puma::prss::{Cursors, SetupStatus};

    fn context() -> Context {
        Context::new([31; 32], [43; 32], 7).unwrap()
    }

    #[test]
    fn fixed_entropy_is_public_role_and_purpose_separated() {
        for (i, party) in [Party::P0, Party::P1, Party::P2].into_iter().enumerate() {
            let mut setup = fixed_setup(context(), party, PrssLimits::default()).unwrap();
            assert_eq!(setup.status(), SetupStatus::Gathering);
            assert_eq!(setup.contribution().unwrap().value(), i as u128 + 1);
            let seeds = setup.seeds.as_ref().unwrap();
            assert_eq!(u128::from_le_bytes(*seeds.private), i as u128 + 101);
            assert_eq!(u128::from_le_bytes(*seeds.own), (i as u128 + 1) * 11);
            assert!(seeds.next.is_none());
            let mut fresh = fixed_setup(context(), party, PrssLimits::default()).unwrap();
            assert_eq!(fresh.contribution().unwrap().value(), i as u128 + 1);
            let fresh_seeds = fresh.seeds.as_ref().unwrap();
            assert_eq!(*seeds.private, *fresh_seeds.private);
            assert_eq!(*seeds.own, *fresh_seeds.own);
            assert!(matches!(setup.finish(), Err(Error::Phase)));
            assert_eq!(setup.status(), SetupStatus::Retired);
        }
    }

    #[test]
    fn fresh_graphs_keep_native_namespace_zero_cursors_and_independent_lifecycle() {
        let mut control = fresh_compound_parties(context(), PrssLimits::default()).unwrap();
        let candidate = fresh_compound_parties(context(), PrssLimits::default()).unwrap();
        for (i, (a, b)) in control.iter().zip(&candidate).enumerate() {
            assert_eq!(a.party() as usize, i);
            assert_eq!(a.party(), b.party());
            assert!(a.matches_context(context()));
            assert!(b.matches_context(context()));
            assert_eq!(a.context(), b.context());
            assert_eq!(a.cursors(), Cursors::default());
            assert_eq!(b.cursors(), Cursors::default());
            assert_eq!(a.status(), multiply::Status::Ready);
            assert_eq!(b.status(), multiply::Status::Ready);
        }
        control[0].cancel();
        assert_eq!(control[0].status(), multiply::Status::Retired);
        assert_eq!(candidate[0].status(), multiply::Status::Ready);
        assert_eq!(candidate[0].cursors(), Cursors::default());
    }

    #[test]
    fn supplied_context_and_limits_are_not_replaced_by_fixture_defaults() {
        let changed = Context::new([32; 32], [44; 32], 8).unwrap();
        let graph = fresh_compound_parties(changed, PrssLimits::default()).unwrap();
        assert!(graph.iter().all(|owner| owner.matches_context(changed)));
        assert!(graph.iter().all(|owner| !owner.matches_context(context())));
        // Zero AES allowance is meaningful for program limits: preserve it,
        // rather than silently upgrading the caller to fixture defaults.
        let limits = PrssLimits::program(1, 0).unwrap();
        let setup = fixed_setup(context(), Party::P0, limits).unwrap();
        assert_eq!(setup.limits, limits);
        let mut graph = fresh_compound_parties(context(), limits).unwrap();
        assert!(graph
            .iter()
            .all(|owner| owner.cursors() == Cursors::default()));
        let unit = multiply::Interval::new(0, 1).unwrap();
        let bounds = multiply::InputBounds {
            left: unit,
            right: unit,
        };
        for owner in &mut graph {
            assert!(matches!(owner.plan(&[bounds]), Err(Error::Bounds)));
            assert_eq!(owner.status(), multiply::Status::Retired);
            assert_eq!(owner.cursors(), Cursors::default());
        }
    }
}
