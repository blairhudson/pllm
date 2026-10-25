use super::*;
#[test]
fn literal_sixteen_group_oracle_not_horner_or_production_helper() {
    let c = std::array::from_fn::<_, 16, _>(|i| (i as f64 - 7.) / 16.);
    for x in [-0.25, 0., 0.125] {
        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;
        let mut b = [0.; 4];
        for i in 0..4 {
            let v = c[4 * i + 1] * x;
            let v = v + c[4 * i + 2] * x2;
            let v = v + x2 * (c[4 * i + 3] * x);
            b[i] = v + c[4 * i];
        }
        let expected = (b[0] + b[1] * x4) + (b[2] + b[3] * x4) * x8;
        assert_eq!(
            stockmeyer(&c, x, &mut Workspace::empty())
                .unwrap()
                .to_bits(),
            expected.to_bits()
        );
    }
    assert_eq!(arithmetic_work(16), Ok(37));
    assert_eq!(arithmetic_work(28), Ok(65));
    assert_eq!(arithmetic_work(32), Ok(74));
}
#[test]
fn shape_nonfinite_and_intermediate_overflow_refuse_no_clipping() {
    let mut w = Workspace::empty();
    assert_eq!(stockmeyer(&[0.; 15], 0., &mut w), Err(Error::Shape));
    assert_eq!(
        stockmeyer(&[0.; 32], f64::MAX, &mut w),
        Err(Error::NonFinite)
    );
    let mut c = [0.; 28];
    c[5] = f64::NAN;
    assert_eq!(stockmeyer(&c, 0., &mut w), Err(Error::NonFinite));
    w.zeroize();
    assert!(w
        .babies
        .iter()
        .chain(&w.first)
        .chain(&w.second)
        .chain(&w.powers)
        .all(|&v| v == 0.));
}
