use super::*;

#[test]
fn scalar_cap_endpoints_strict_benefit_and_overflow() {
    assert_eq!(cap_energy(0, 9), Err(Error::Shape));
    assert_eq!(cap_energy(3, -1), Err(Error::Bounds));
    assert_eq!(cap_energy(3, 0).unwrap(), 0);
    assert_eq!(cap_energy(3, 9).unwrap(), 243);
    assert_eq!(strict_improvement(150, 243), Some(150));
    assert_eq!(strict_improvement(242, 243), Some(242));
    assert_eq!(strict_improvement(243, 243), None);
    assert_eq!(strict_improvement(244, 243), None);
    assert_eq!(strict_improvement(0, 0), None);
    assert_eq!(strict_improvement(0, 1), Some(0));
    let square = (i64::MAX as u128).pow(2);
    assert_eq!(cap_energy(4, i64::MAX).unwrap(), square * 4);
    assert_eq!(cap_energy(5, i64::MAX), Err(Error::Bounds));
    assert_eq!(strict_improvement(u128::MAX, u128::MAX), None);
    assert_eq!(
        strict_improvement(u128::MAX - 1, u128::MAX),
        Some(u128::MAX - 1)
    );
}

#[test]
fn independent_exhaustive_zero_mux_including_rejected_rows() {
    let mut rejected_with_strict_benefit = 0;
    for cap in 0_i64..=4 {
        let scalar = cap_energy(3, cap).unwrap();
        for a in -5_i64..=5 {
            for b in -5_i64..=5 {
                for c in -5_i64..=5 {
                    let row = [a, b, c];
                    let original: u128 = row
                        .iter()
                        .map(|x| u128::from(x.unsigned_abs()).pow(2))
                        .sum();
                    let safe = row.map(|x| if (-cap..=cap).contains(&x) { x } else { 0 });
                    let actual: u128 = safe
                        .iter()
                        .map(|x| u128::from(x.unsigned_abs()).pow(2))
                        .sum();
                    // Enumerate loose as well as exact true producer bounds.
                    for bound in original..=75 {
                        let retained = strict_improvement(bound, scalar);
                        assert!(actual <= retained.unwrap_or(scalar));
                        assert!(actual <= bound.min(scalar));
                        if retained.is_some() && row != safe {
                            rejected_with_strict_benefit += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(rejected_with_strict_benefit > 0);
}
