use super::*;
use std::collections::BTreeSet;

const UNIT: i64 = 1 << 18;

fn prove<'a>(row: &[i64], gamma: &'a [i64]) -> ConditionalRmsEnergy<'a> {
    let energy = row
        .iter()
        .map(|x| u128::from(x.unsigned_abs()).pow(2))
        .sum();
    let input = Interval::new(*row.iter().min().unwrap(), *row.iter().max().unwrap()).unwrap();
    let p = RmsEnergyParameters::new(1, row.len(), input, gamma, energy).unwrap();
    let r = p.resources().unwrap();
    let mut workspace = RmsEnergyWorkspace::new(&p, r.limits()).unwrap();
    p.prove(&mut workspace).unwrap()
}

// Independent frozen INTEGER equations, not ideal-real normalization or native
// interval/rsqrt helpers. Enumerate floor/ceil at each stochastic boundary.
fn choices(x: i128) -> Vec<i64> {
    let q = i128::from(UNIT);
    (x.div_euclid(q)..=-(-x).div_euclid(q))
        .map(|v| i64::try_from(v).unwrap())
        .collect()
}
fn inverses(d: i64) -> BTreeSet<i64> {
    let j = (d as u64).ilog2();
    let factor = 1_i128 << (34 - j);
    let correction = (512 + 212 * i128::from((j + 1) % 2 == 0)) * (1_i128 << (17 - (j + 1) / 2));
    let mut out = BTreeSet::new();
    for u in choices(i128::from(d) * factor) {
        for u2 in choices(i128::from(u) * i128::from(u)) {
            for u3 in choices(i128::from(u2) * i128::from(u)) {
                for u4 in choices(i128::from(u3) * i128::from(u)) {
                    let polynomial = -4_057_974 * i128::from(u) + 10_085_068 * i128::from(u2)
                        - 13_072_088 * i128::from(u3)
                        + 6_823_457 * i128::from(u4);
                    for scaled in choices(polynomial) {
                        out.extend(choices(i128::from(scaled + 1_086_023) * correction));
                    }
                }
            }
        }
    }
    out
}
fn check_integer_oracle(row: &[i64], gamma: &[i64], bound: u128) -> usize {
    // Cartesian sums by value, no theorem or native helper supplies expected.
    let mut sums = BTreeSet::from([0_i128]);
    for &x in row {
        sums = sums
            .iter()
            .flat_map(|s| {
                choices(i128::from(x) * i128::from(x))
                    .into_iter()
                    .map(move |v| s + i128::from(v))
            })
            .collect();
    }
    let c = ((1.0_f32 / row.len() as f32) * UNIT as f32) as i64;
    let mut checked = 0;
    for sum in sums {
        for mean in choices(sum * i128::from(c)) {
            for inverse in inverses(mean + 1) {
                // Independent maximum of each coordinate: all combinations lie
                // below sum of maxima, including negative gamma/carries.
                let mut energy = 0;
                for (&x, &g) in row.iter().zip(gamma) {
                    let maximum = choices(i128::from(x) * i128::from(inverse))
                        .into_iter()
                        .flat_map(|n| choices(i128::from(n) * i128::from(g)))
                        .map(|p| u128::from(p.unsigned_abs()).pow(2))
                        .max()
                        .unwrap();
                    energy += maximum;
                }
                assert!(
                    energy <= bound,
                    "row={row:?}, gamma={gamma:?}, d={}, energy={energy}, bound={bound}",
                    mean + 1
                );
                checked += 1;
                assert!(checked <= 100_000);
            }
        }
    }
    checked
}

#[test]
fn frozen_integer_oracle_preserves_order_gamma_mean_and_all_rounding() {
    let mut checked = 0;
    for row in [[-2, 1, 0], [1, -1, 1], [0, 0, 0], [2, 2, -2]] {
        for gamma in [[UNIT, -UNIT, UNIT / 7], [0, UNIT / 3, -UNIT]] {
            let theorem = prove(&row, &gamma);
            checked += check_integer_oracle(&row, &gamma, theorem.conditional_squared_upper());
        }
    }
    assert!(checked > 8);
    assert_eq!(numeric::rms_mean_coefficient(3).unwrap(), 87381);
}

#[test]
fn exact_zero_energy_and_zero_gamma_do_not_use_ideal_real_rms() {
    assert_eq!(
        prove(&[0, 0], &[UNIT, -UNIT]).conditional_squared_upper(),
        0
    );
    assert_eq!(prove(&[2, -2], &[0, 0]).conditional_squared_upper(), 0);
    let x = ((2 * UNIT - 1) as u128).isqrt() as i64;
    let gamma = [UNIT; 3];
    let theorem = prove(&[x; 3], &gamma);
    assert!(check_integer_oracle(&[x; 3], &gamma, theorem.conditional_squared_upper()) > 0);
}

#[test]
fn preconstruction_short_caps_and_overflow_refuse_without_native_factory() {
    let gamma = [UNIT; 2];
    let p = RmsEnergyParameters::new(1, 2, Interval::new(-2, 2).unwrap(), &gamma, 8).unwrap();
    let r = p.resources().unwrap();
    let before = rms_work::attempts();
    for field in 0..3 {
        let mut limits = r.limits();
        match field {
            0 => limits.known_bytes -= 1,
            1 => limits.denominator_cases -= 1,
            _ => limits.polynomial_cells -= 1,
        }
        assert!(matches!(
            RmsEnergyWorkspace::new(&p, limits),
            Err(Error::Bounds)
        ));
    }
    let invalid = RmsEnergyParameters::new(1, 2, Interval::new(1, 1).unwrap(), &gamma, 0);
    assert!(matches!(invalid, Err(Error::Bounds)));
    let overflow = RmsEnergyParameters::new(1, 2, Interval::new(-1, 1).unwrap(), &gamma, u128::MAX);
    // A huge loose premise may be intersected with the interval, not wrapped.
    if let Ok(p) = overflow {
        let r = p.resources().unwrap();
        let mut workspace = RmsEnergyWorkspace::new(&p, r.limits()).unwrap();
        // Preserve the original native validator's checked raw-premise overflow,
        // even though the conditional interval domain itself can be intersected.
        assert!(matches!(p.prove(&mut workspace), Err(Error::Bounds)));
        assert!(workspace.is_poisoned());
    }
    assert_eq!(rms_work::attempts(), before);
}

fn admitted<'a>(gamma: &'a [i64], total: &mut WorkBudget) -> AdmittedFactory<'a> {
    let factory = Construction::RmsNormWithInputNorm {
        rows: 1,
        width: gamma.len(),
        gamma_len: gamma.len(),
    }
    .bind(FactoryParameters::RmsNormWithInputNorm {
        input: Interval::new(-2, 2).unwrap(),
        gamma,
        squared_norm: 8,
    })
    .unwrap();
    let mut work = WorkPreflightWorkspace::empty();
    let ticket = factory.preflight(1, &mut work).unwrap();
    let mut ledger = CallerWorkLedger::new(total, u64::MAX, u64::MAX).unwrap();
    let (part, tail) = ledger.admit(ticket).unwrap().into_parts();
    assert!(tail.is_none());
    part
}

#[test]
fn original_factory_ticket_pins_gamma_native_arc_work_and_legacy_program() {
    let gamma = [UNIT, -UNIT];
    let p = RmsEnergyParameters::new(1, 2, Interval::new(-2, 2).unwrap(), &gamma, 8).unwrap();
    let mut ws = RmsEnergyWorkspace::new(&p, p.resources().unwrap().limits()).unwrap();
    let mut total = WorkBudget::default();
    let owner = admitted(&gamma, &mut total)
        .prepare_rms_energy(&mut ws)
        .unwrap()
        .build()
        .unwrap();
    let (legacy, certificate) =
        Program::rms_norm_with_input_norm(1, 2, Interval::new(-2, 2).unwrap(), &gamma, 8).unwrap();
    assert_eq!(owner.program().digest(), legacy.digest());
    assert_eq!(owner.program().work_budget(), legacy.work_budget());
    assert_eq!(total, legacy.work_budget());
    assert_eq!(
        owner.original_certificate(),
        ProducedCertificate::GeneralOutputRowSquared(certificate)
    );
    let clone = Arc::clone(owner.program());
    assert!(owner.check_native(&clone).is_ok());
    assert!(owner.check_gamma(&gamma).is_ok());
    let foreign_gamma = Box::new(gamma);
    assert!(matches!(
        owner.check_gamma(foreign_gamma.as_ref()),
        Err(Error::Binding)
    ));
    assert!(matches!(
        owner.check_native(&Arc::new(legacy)),
        Err(Error::Binding)
    ));
    let weak = Arc::downgrade(owner.program());
    drop(clone);
    drop(owner);
    assert!(weak.upgrade().is_none());
}

#[test]
fn work_short_and_attention_factory_cannot_enter_rms_constructor() {
    let gamma = [UNIT; 2];
    let factory = Construction::RmsNormWithInputNorm {
        rows: 1,
        width: 2,
        gamma_len: 2,
    }
    .bind(FactoryParameters::RmsNormWithInputNorm {
        input: Interval::new(-2, 2).unwrap(),
        gamma: &gamma,
        squared_norm: 8,
    })
    .unwrap();
    let mut workspace = WorkPreflightWorkspace::empty();
    let ticket = factory.preflight(1, &mut workspace).unwrap();
    let expected = ticket.summary().work;
    let before = rms_work::attempts();
    let mut total = WorkBudget::default();
    let mut ledger =
        CallerWorkLedger::new(&mut total, expected.operations() - 1, u64::MAX).unwrap();
    assert!(ledger.admit(ticket).is_err());
    assert_eq!(total, WorkBudget::default());
    let factory = Construction::SoftmaxCertified { rows: 1, width: 2 }
        .bind(FactoryParameters::SoftmaxCertified {
            input: Interval::new(-1, 0).unwrap(),
        })
        .unwrap();
    let ticket = factory.preflight(1, &mut workspace).unwrap();
    let (part, _) = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX)
        .unwrap()
        .admit(ticket)
        .unwrap()
        .into_parts();
    let p = RmsEnergyParameters::new(1, 2, Interval::new(-2, 2).unwrap(), &gamma, 8).unwrap();
    let mut ws = RmsEnergyWorkspace::new(&p, p.resources().unwrap().limits()).unwrap();
    assert!(matches!(
        part.prepare_rms_energy(&mut ws),
        Err(Error::Binding)
    ));
    assert_eq!(rms_work::attempts(), before);
}

#[test]
fn error_poison_and_named_workspace_capacity_remain_explicit() {
    let gamma = [UNIT; 2];
    let p = RmsEnergyParameters::new(1, 2, Interval::new(-2, 2).unwrap(), &gamma, 8).unwrap();
    let r = p.resources().unwrap();
    assert!(
        r.known_bytes
            >= 2 * size_of::<RmsEnergyWorkspace>()
                + size_of::<Result<RmsEnergyWorkspace, Error>>()
                + size_of::<RmsWorkWorkspace>()
                + size_of::<SoftmaxWorkWorkspace>()
    );
    assert!(r.denominator_cases <= 35 * 8);
    let mut ws = RmsEnergyWorkspace::new(&p, r.limits()).unwrap();
    ws.limits.denominator_cases = 0;
    assert!(matches!(p.prove(&mut ws), Err(Error::Bounds)));
    assert!(ws.is_poisoned());
    let p = RmsEnergyParameters::new(1, 2, Interval::new(-2, 2).unwrap(), &gamma, 8).unwrap();
    assert!(matches!(p.prove(&mut ws), Err(Error::Phase)));
    ws.reset();
    assert!(!ws.is_poisoned());
}
