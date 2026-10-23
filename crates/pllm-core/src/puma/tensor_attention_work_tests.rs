//! Independent pre-extraction softmax composition and native work comparisons.
//! No new profile/golden, runtime execution, RNG or caller-proof authority.
use super::attention_work::factory_attempts;
use super::*;

#[derive(Clone, Copy, Debug)]
enum Factory {
    Original,
    Guarded,
    Certified,
}

impl Factory {
    fn work(
        self,
        rows: usize,
        width: usize,
        input: Interval,
        workspace: &mut SoftmaxWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        match self {
            Self::Original => Program::softmax_work(rows, width, input, workspace),
            Self::Guarded => Program::softmax_guarded_work(rows, width, input, workspace),
            Self::Certified => Program::softmax_certified_work(rows, width, input, workspace),
        }
    }
    fn construct(
        self,
        rows: usize,
        width: usize,
        input: Interval,
    ) -> Result<(Program, Option<u64>), Error> {
        match self {
            Self::Original => Program::softmax(rows, width, input).map(|p| (p, None)),
            Self::Guarded => Program::softmax_guarded(rows, width, input).map(|p| (p, None)),
            Self::Certified => {
                Program::softmax_certified(rows, width, input).map(|(p, m)| (p, Some(m)))
            }
        }
    }
}

fn keep(b: &mut Builder, z: Value) -> Result<Value, Error> {
    let threshold = b.affine(
        z,
        None,
        Affine {
            left: -1,
            right: 0,
            constant: -14 * (1 << 18),
        },
    )?;
    let keep = b.sign(threshold)?;
    b.from_bit(keep)
}

// Literal old factory, not softmax_recipe/SoftmaxSink/any work-query path.
// Original reciprocal numeric primitive and public mass proof remain owners of
// those subgraphs; the softmax-level order/refinements are independent here.
fn original(
    factory: Factory,
    rows: usize,
    width: usize,
    input: Interval,
) -> Result<(Program, Option<u64>), Error> {
    let one = 1 << 18;
    let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
    if width == 0 || rows == 0 {
        return Err(Error::Bounds);
    }
    let mut b = Builder::row(lanes, input)?;
    let x = b.input();
    let mut maximum = x;
    let mut offset = 1;
    while offset < width {
        let other = b.row_map(maximum, width, RowMap::Rotate(offset))?;
        let difference = b.affine(
            maximum,
            Some(other),
            Affine {
                left: 1,
                right: -1,
                constant: 0,
            },
        )?;
        let bit = b.sign(difference)?;
        let bit = b.from_bit(bit)?;
        let difference = b.affine(
            other,
            Some(maximum),
            Affine {
                left: 1,
                right: -1,
                constant: 0,
            },
        )?;
        let chosen = b.product(bit, difference, false)?;
        maximum = b.affine(
            maximum,
            Some(chosen),
            Affine {
                left: 1,
                right: 1,
                constant: 0,
            },
        )?;
        b.0.nodes[maximum.0 as usize].as_mut().unwrap().bounds = input;
        offset *= 2;
    }
    let z = b.affine(
        x,
        Some(maximum),
        Affine {
            left: 1,
            right: -1,
            constant: 0,
        },
    )?;
    let lower = input
        .lower()
        .checked_sub(input.upper())
        .ok_or(Error::Bounds)?;
    b.0.nodes[z.0 as usize].as_mut().unwrap().bounds = Interval::new(lower, 0)?;
    let guarded = !matches!(factory, Factory::Original);
    let retained = if guarded {
        Some(keep(&mut b, z)?)
    } else {
        None
    };
    let safe = if let Some(retained) = retained {
        let safe = b.product(z, retained, false)?;
        b.0.nodes[safe.0 as usize].as_mut().unwrap().bounds =
            Interval::new(lower.max(-14 * one + 1), 0)?;
        safe
    } else {
        z
    };
    let reduced = b.rescale(safe, 5)?;
    let mut exp = b.affine(
        reduced,
        None,
        Affine {
            left: 1,
            right: 0,
            constant: one,
        },
    )?;
    for _ in 0..5 {
        exp = b.square(exp)?;
    }
    let retained = match retained {
        Some(v) => v,
        None => keep(&mut b, z)?,
    };
    let exp = b.product(exp, retained, false)?;
    let total = b.row_map(exp, width, RowMap::Sum)?;
    let range = Interval::new(one, b.node(total, false)?.bounds.upper())?;
    b.0.nodes[total.0 as usize].as_mut().unwrap().bounds = range;
    let inverse = numeric::reciprocal(&mut b, total, range)?;
    let mass = if matches!(factory, Factory::Certified) {
        let (range, product) = numeric::reciprocal_mass_bounds(range)?;
        b.0.nodes[inverse.0 as usize].as_mut().unwrap().bounds = range;
        Some(
            u64::try_from(
                (i128::from(product.upper()) + width as i128 * i128::from(one - 1))
                    / i128::from(one),
            )
            .map_err(|_| Error::Bounds)?,
        )
    } else {
        None
    };
    let inverse = b.row_map(inverse, width, RowMap::Broadcast)?;
    let output = b.product(exp, inverse, true)?;
    Ok((b.finish(output)?, mass))
}

fn same_program(a: &Program, e: &Program) {
    assert_eq!(a.nodes, e.nodes);
    assert_eq!(a.len, e.len);
    assert_eq!(a.bit_widths, e.bit_widths);
    assert_eq!(a.digest(), e.digest());
    assert_eq!(a.numeric_profile(), e.numeric_profile());
    assert_eq!(a.last_use, e.last_use);
    assert_eq!(a.slots, e.slots);
    assert_eq!(a.register_count, e.register_count);
    assert_eq!(a.work_budget(), e.work_budget());
    assert_eq!(a.public_affine_count, e.public_affine_count);
    assert!(a.public_affines.iter().all(|table| table.capacity() == 0));
}

#[test]
fn softmax_work_matches_native_and_literal_instruction_vectors_domains_and_widths() {
    let mut workspace = SoftmaxWorkWorkspace::empty();
    let mut successful = [0; 3];
    for (index, factory) in [Factory::Original, Factory::Guarded, Factory::Certified]
        .into_iter()
        .enumerate()
    {
        for rows in [1, 2] {
            for width in [1, 2, 3, 7, 16, 127, 128, 256, 512, MAX_ROW_LANES] {
                for (lo, hi) in [
                    (0, 0),
                    (-32, 32),
                    (-(1 << 18), 1 << 18),
                    (-14 * (1 << 18), 0),
                    (-15 * (1 << 18), 1 << 18),
                    (i64::MIN, i64::MAX),
                ] {
                    let input = Interval::new(lo, hi).unwrap();
                    let attempts = factory_attempts();
                    let quote = factory.work(rows, width, input, &mut workspace);
                    assert_eq!(factory_attempts(), attempts, "{factory:?}");
                    let actual = factory.construct(rows, width, input);
                    let expected = original(factory, rows, width, input);
                    match (quote,actual,expected) {
                    (Ok(q),Ok((a,am)),Ok((e,em)))=>{workspace.assert_native_metadata(&a);successful[index]+=1;assert_eq!(q,a.work_budget());assert_eq!(q.cursors(),e.work_budget().cursors());assert_eq!(am,em);same_program(&a,&e);},
                    (Err(q),Err(a),Err(e))=>{assert_eq!(q,a);assert_eq!(a,e);},
                    _=>panic!("work/native/literal outcome differs: {factory:?}, {rows}x{width}, [{lo},{hi}]"),
                }
                }
            }
        }
    }
    assert!(successful.into_iter().all(|count| count > 0));
}

#[test]
fn softmax_work_invalid_geometry_extreme_domains_and_workspace_reuse_match_error_precedence() {
    let mut reused = SoftmaxWorkWorkspace::empty();
    for factory in [Factory::Original, Factory::Guarded, Factory::Certified] {
        for (rows, width) in [
            (0, 0),
            (0, usize::MAX),
            (usize::MAX, 2),
            (2, usize::MAX),
            (1, MAX_ROW_LANES + 1),
            (1, 2),
        ] {
            let extreme = Interval::new(i64::MIN, i64::MAX).unwrap();
            let attempts = factory_attempts();
            let quote = factory.work(rows, width, extreme, &mut reused);
            assert!(quote.is_err());
            assert_eq!(factory_attempts(), attempts);
            assert_eq!(
                quote,
                factory
                    .construct(rows, width, extreme)
                    .map(|(p, _)| p.work_budget())
            );
            let input = Interval::new(0, 0).unwrap();
            let mut fresh = SoftmaxWorkWorkspace::empty();
            assert_eq!(
                factory.work(1, 2, input, &mut reused),
                factory.work(1, 2, input, &mut fresh)
            );
        }
    }
}

fn full_tail_work(
    factory: Factory,
    input: Interval,
    space: &mut SoftmaxWorkWorkspace,
) -> Result<WorkBudget, Error> {
    factory
        .work(2, 3, input, space)?
        .checked_mul(2)?
        .checked_add(factory.work(1, 3, input, space)?)
}

fn admitted(
    factory: Factory,
    input: Interval,
    space: &mut SoftmaxWorkWorkspace,
    ops: u64,
    aes: u64,
) -> Result<(), Error> {
    let q = full_tail_work(factory, input, space)?;
    q.admit(0, crate::puma::prss::Cursors::default(), ops, aes)?;
    factory.construct(2, 3, input)?;
    factory.construct(1, 3, input)?;
    Ok(())
}

#[test]
fn softmax_work_full_tail_caps_refuse_before_factory_entry_and_keep_call_reset_rounding() {
    let mut space = SoftmaxWorkWorkspace::empty();
    let input = Interval::new(0, 0).unwrap();
    for factory in [Factory::Original, Factory::Guarded, Factory::Certified] {
        let attempts = factory_attempts();
        let q = full_tail_work(factory, input, &mut space).unwrap();
        assert_eq!(factory_attempts(), attempts);
        let (ops, aes) = (q.operations(), q.cursors().aes_work);
        for (ops, aes) in [(0, aes), (ops - 1, aes), (ops, 0), (ops, aes - 1)] {
            assert!(matches!(
                admitted(factory, input, &mut space, ops, aes),
                Err(Error::Bounds)
            ));
            assert_eq!(factory_attempts(), attempts);
        }
        admitted(factory, input, &mut space, ops, aes).unwrap();
        assert_eq!(factory_attempts(), attempts + 2);
        let full = factory.construct(2, 3, input).unwrap().0.work_budget();
        let tail = factory.construct(1, 3, input).unwrap().0.work_budget();
        assert_eq!(q, full.checked_mul(2).unwrap().checked_add(tail).unwrap());
        // Five rows in one fill do not have the two full + one tail call-reset
        // PRSS rounding of the original invocation schedule.
        let merged = factory.work(5, 3, input, &mut space).unwrap();
        assert_ne!(q.cursors(), merged.cursors());
    }
}

#[test]
fn softmax_work_checked_request_aggregation_and_cursor_overflow_never_enter_factories() {
    let mut space = SoftmaxWorkWorkspace::empty();
    let input = Interval::new(0, 0).unwrap();
    let attempts = factory_attempts();
    let q = Program::softmax_certified_work(1, 2, input, &mut space).unwrap();
    assert!(matches!(q.checked_mul(u64::MAX), Err(Error::Bounds)));
    assert!(matches!(
        WorkBudget::local()
            .checked_mul(u64::MAX)
            .unwrap()
            .checked_add(q),
        Err(Error::Bounds)
    ));
    let c = q.cursors();
    for prior in [
        crate::puma::prss::Cursors {
            public_blocks: u64::MAX,
            ..Default::default()
        },
        crate::puma::prss::Cursors {
            pair_blocks: u64::MAX,
            ..Default::default()
        },
        crate::puma::prss::Cursors {
            private_blocks: u64::MAX,
            ..Default::default()
        },
        crate::puma::prss::Cursors {
            aes_work: u64::MAX,
            ..Default::default()
        },
    ] {
        assert!(c.public_blocks > 0 && c.pair_blocks > 0 && c.private_blocks > 0 && c.aes_work > 0);
        assert!(matches!(
            q.admit(0, prior, u64::MAX, u64::MAX),
            Err(Error::Bounds)
        ));
    }
    assert_eq!(factory_attempts(), attempts);
}

#[test]
fn softmax_workspace_quote_precedes_caller_construction_without_granting_budget_authority() {
    fn workspace(limit: usize, calls: &mut usize) -> Result<SoftmaxWorkWorkspace, Error> {
        if SoftmaxWorkWorkspace::workspace_bytes() > limit {
            return Err(Error::Bounds);
        }
        *calls += 1;
        Ok(SoftmaxWorkWorkspace::empty())
    }
    let bytes = SoftmaxWorkWorkspace::workspace_bytes();
    assert_eq!(bytes, std::mem::size_of::<SoftmaxWorkWorkspace>());
    assert!(
        bytes
            >= std::mem::size_of::<[Option<Node>; MAX_NODES]>()
                + std::mem::size_of::<[u8; MAX_NODES]>()
                + 2 * std::mem::size_of::<usize>()
    );
    let mut calls = 0;
    assert!(workspace(bytes - 1, &mut calls).is_err());
    assert_eq!(calls, 0);
    let mut space = workspace(bytes, &mut calls).unwrap();
    assert_eq!(calls, 1);
    assert!(Program::softmax_work(1, 2, Interval::new(0, 0).unwrap(), &mut space).is_ok());
    // No assertion about private proof temporaries, native/dependency stacks,
    // allocator/process RSS or a complete constructor follows from this quote.
}

#[test]
fn shared_native_raw_product_still_accepts_full_ring64_and_keeps_binding_precedence() {
    let input = Interval::new(1 << 30, 1 << 30).unwrap();
    let mut b = Builder::new(1, input).unwrap();
    let raw = b.product(b.input(), b.input(), false).unwrap();
    assert_eq!(
        b.node(raw, false).unwrap().bounds,
        Interval::new(1 << 60, 1 << 60).unwrap()
    );
    assert_eq!(
        b.finish(raw).unwrap().work_budget(),
        WorkBudget::product(1, false).unwrap()
    );
    let mut b = Builder::new(1, input).unwrap();
    // An invalid input wins before the invalid coefficient, just as originally.
    assert!(matches!(
        b.affine(
            Value(u8::MAX),
            None,
            Affine {
                left: 1,
                right: 1,
                constant: 0
            }
        ),
        Err(Error::Binding)
    ));
    assert!(matches!(
        b.affine(
            b.input(),
            None,
            Affine {
                left: 1,
                right: 1,
                constant: 0
            }
        ),
        Err(Error::Bounds)
    ));
}
