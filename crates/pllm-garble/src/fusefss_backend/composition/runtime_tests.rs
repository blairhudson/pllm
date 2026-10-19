use super::*;
use crate::fusefss_backend::MAX_ALLOCATION_BYTES;
use std::sync::atomic::Ordering;

fn context() -> Context {
    Context {
        plan: [4; 32],
        session: [5; 32],
        operation: [6; 32],
        first_tensor_index: 17,
    }
}
fn builder<'a>(bits: &[u8], budget: &'a Budget) -> Builder<'a> {
    Builder::new([7; 32], b"typed-test-contract", bits, budget).unwrap()
}

// Only this harness reconstructs. Dealer is external to both runtimes and
// issues only currently requested public step, never a whole-plan key bundle.
fn execute<'a>(plan: &Plan<'a>, inputs: [&[u64]; 2]) -> Result<[Output<'a>; 2], Error> {
    execute_counted(plan, inputs).map(|(outputs, _, _)| outputs)
}
fn execute_counted<'a>(
    plan: &Plan<'a>,
    inputs: [&[u64]; 2],
) -> Result<([Output<'a>; 2], usize, usize), Error> {
    let mut a = plan.runtime(0, context(), inputs[0])?;
    let mut b = plan.runtime(1, context(), inputs[1])?;
    let mut events = (a.poll()?, b.poll()?);
    let (mut frames, mut bytes) = (0, 0);
    loop {
        events = match events {
            (Event::NeedMaterial { pc: x }, Event::NeedMaterial { pc: y }) => {
                assert_eq!(x, y);
                let [ma, mb] = plan.issue_step(x, context())?;
                (a.feed(ma)?, b.feed(mb)?)
            }
            (Event::Frame(fa), Event::Frame(fb)) => {
                frames += 2;
                bytes += fa.as_bytes().len() + fb.as_bytes().len();
                (a.advance(fb.as_bytes())?, b.advance(fa.as_bytes())?)
            }
            (Event::Output(oa), Event::Output(ob)) => {
                assert!(a.is_consumed() && b.is_consumed());
                return Ok(([oa, ob], frames, bytes));
            }
            _ => panic!("public schedules diverged"),
        };
    }
}
fn reconstruct(plan: &Plan<'_>, pair: &[Output<'_>; 2]) -> Vec<u64> {
    pair[0]
        .shares()
        .iter()
        .zip(pair[1].shares())
        .enumerate()
        .map(|(i, (&a, &b))| {
            if plan.output_is_xor(i).unwrap() {
                a ^ b
            } else {
                a.wrapping_add(b) & mask(plan.output_bits(i).unwrap())
            }
        })
        .collect()
}

#[test]
fn local_ring_widths_constants_and_native_output_leases() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut b = builder(&[8, 8], &budget);
    let x = b.input(0).unwrap();
    let y = b.input(1).unwrap();
    let add = b.add(x, y).unwrap();
    let sub = b.sub(x, y).unwrap();
    let product = b.mul_public(sub, u64::MAX).unwrap();
    let narrow = b.narrow(add, 3).unwrap();
    let constant = b.constant(8, 193).unwrap();
    assert_eq!(b.add(x, narrow), Err(Error::Domain));
    assert_eq!(b.narrow(x, 9), Err(Error::Domain));
    let p = b.finish(&[add, sub, product, narrow, constant]).unwrap();
    let out = execute(&p, [&[211, 8], &[100, 250]]).unwrap();
    assert_eq!(reconstruct(&p, &out), [57, 53, 203, 1, 193]);
    assert_eq!(out[0].shares()[4], 193);
    assert_eq!(out[1].shares()[4], 0);
    drop(p);
    assert_eq!(budget.used_bytes(), 2 * 5 * 8);
    drop(out);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn mixed_width_protected_extensions_and_exact_products() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut b = builder(&[5], &budget);
    let x = b.input(0).unwrap();
    let z = b.convert(x, conversions::Operation::ZeroExtend, 9).unwrap();
    let s = b
        .convert(
            x,
            conversions::Operation::SignExtend {
                range: conversions::SignedRange::full_ring(5).unwrap(),
            },
            9,
        )
        .unwrap();
    let t = b
        .convert(
            s,
            conversions::Operation::Tr {
                shift: 2,
                range: conversions::SignedRange::full_ring(9).unwrap(),
            },
            7,
        )
        .unwrap();
    let a = b
        .convert(
            s,
            conversions::Operation::Ars {
                shift: 2,
                range: conversions::SignedRange::full_ring(9).unwrap(),
            },
            9,
        )
        .unwrap();
    let square = b.multiply(s, s).unwrap();
    let p = b.finish(&[z, s, t, a, square]).unwrap();
    for value in 0..32u64 {
        let share = 19;
        let other = value.wrapping_sub(share) & 31;
        let pair = execute(&p, [&[share], &[other]]).unwrap();
        let signed = if value >= 16 {
            value as i64 - 32
        } else {
            value as i64
        };
        assert_eq!(
            reconstruct(&p, &pair),
            [
                value,
                signed as u64 & 511,
                (signed >> 2) as u64 & 127,
                (signed >> 2) as u64 & 511,
                (signed * signed) as u64 & 511
            ]
        );
    }
    assert_eq!(budget.used_bytes(), p.resources().plan_reservation_bytes);
}

#[test]
fn comparisons_return_arithmetic_bits_and_select_requires_provenance() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut b = builder(&[5], &budget);
    let x = b.input(0).unwrap();
    let yes = b.constant(5, 13).unwrap();
    let no = b.constant(5, 27).unwrap();
    assert_eq!(b.select(x, yes, no), Err(Error::Domain));
    assert_eq!(
        b.convert(x, conversions::Operation::BooleanToArithmetic, 5),
        Err(Error::Domain)
    );
    let slt = b.lt_signed(x, -3).unwrap();
    let ult = b.lt_unsigned(x, 19).unwrap();
    let selected = b.select(slt, yes, no).unwrap();
    let forged = b.add(slt, ult).unwrap();
    assert_eq!(b.select(forged, yes, no), Err(Error::Domain));
    let p = b.finish(&[slt, ult, selected]).unwrap();
    assert_eq!(p.resources().secret_steps, 5);
    for value in 0..32u64 {
        let pair = execute(&p, [&[29], &[value.wrapping_sub(29) & 31]]).unwrap();
        let signed = if value >= 16 {
            value as i64 - 32
        } else {
            value as i64
        };
        assert_eq!(
            reconstruct(&p, &pair),
            [
                u64::from(signed < -3),
                u64::from(value < 19),
                if signed < -3 { 13 } else { 27 }
            ]
        );
    }
}

#[test]
fn comparison_edges_include_one_bit_and_fifty_bit_native_rings() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    for bits in [1, 2, 7, 50] {
        let mut b = builder(&[bits], &budget);
        let x = b.input(0).unwrap();
        let minimum = -(1i64 << (bits - 1));
        let negative = b.lt_signed(x, 0).unwrap();
        let never_signed = b.lt_signed(x, minimum).unwrap();
        let never_unsigned = b.lt_unsigned(x, 0).unwrap();
        let always_unsigned = b.lt_unsigned(x, 1u64 << bits).unwrap();
        let p = b
            .finish(&[negative, never_signed, never_unsigned, always_unsigned])
            .unwrap();
        for value in [
            0,
            1,
            (1u64 << (bits - 1)) - 1,
            1u64 << (bits - 1),
            mask(bits),
        ] {
            let share = mask(bits) - 1;
            let pair = execute(&p, [&[share], &[value.wrapping_sub(share) & mask(bits)]]).unwrap();
            assert_eq!(
                reconstruct(&p, &pair),
                [u64::from(value >= 1u64 << (bits - 1)), 0, 0, 1]
            );
        }
    }
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn lookup_is_existing_zero_extended_table_and_composes() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut b = builder(&[4], &budget);
    let x = b.input(0).unwrap();
    let table = b.table(4, &[3, 15, 2, 11]).unwrap();
    let y = b.lookup(x, table).unwrap();
    let product = b.multiply(x, y).unwrap();
    let p = b.finish(&[y, product]).unwrap();
    for x in 0..16u64 {
        let pair = execute(&p, [&[12], &[x.wrapping_sub(12) & 15]]).unwrap();
        let y = [3, 15, 2, 11].get(x as usize).copied().unwrap_or(0);
        assert_eq!(reconstruct(&p, &pair), [y, (x * y) & 15]);
    }
}

#[test]
fn wrong_role_context_order_and_cancel_burn() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut b = builder(&[5], &budget);
    let x = b.input(0).unwrap();
    let y = b.multiply(x, x).unwrap();
    let z = b.multiply(y, x).unwrap();
    let p = b.finish(&[z]).unwrap();
    {
        let mut r = p.runtime(0, context(), &[12]).unwrap();
        let [a, wrong_role] = p.issue_step(0, context()).unwrap();
        drop(a);
        assert!(matches!(r.feed(wrong_role), Err(Error::Context)));
        assert!(r.is_consumed());
    }
    {
        let mut r = p.runtime(0, context(), &[12]).unwrap();
        let [wrong_pc, other] = p.issue_step(1, context()).unwrap();
        drop(other);
        assert!(matches!(r.feed(wrong_pc), Err(Error::Context)));
        assert!(r.is_consumed());
    }
    {
        let mut r = p.runtime(0, context(), &[12]).unwrap();
        let mut different = context();
        different.session[0] ^= 1;
        let [wrong_context, other] = p.issue_step(0, different).unwrap();
        drop(other);
        assert!(matches!(r.feed(wrong_context), Err(Error::Context)));
        assert!(r.is_consumed());
    }
    {
        let mut r = p.runtime(0, context(), &[12]).unwrap();
        assert!(matches!(r.advance(&[]), Err(Error::Context)));
        assert!(r.is_consumed());
    }
    {
        let mut r = p.runtime(0, context(), &[12]).unwrap();
        let [a, other] = p.issue_step(0, context()).unwrap();
        drop(other);
        let frame = r.feed(a).unwrap();
        assert!(matches!(r.poll(), Err(Error::Context)));
        assert!(r.is_consumed());
        drop(frame);
    }
    {
        let mut r = p.runtime(0, context(), &[12]).unwrap();
        r.cancel();
        assert!(matches!(r.poll(), Err(Error::Consumed)));
    }
    {
        let mut other = Builder::new([7; 32], b"different-plan", &[5], &budget).unwrap();
        let x = other.input(0).unwrap();
        let y = other.multiply(x, x).unwrap();
        let other = other.finish(&[y]).unwrap();
        let [wrong_plan, unused] = other.issue_step(0, context()).unwrap();
        drop(unused);
        let mut r = p.runtime(0, context(), &[12]).unwrap();
        assert!(matches!(r.feed(wrong_plan), Err(Error::Context)));
        assert!(r.is_consumed());
    }
    {
        let mut r = p.runtime(0, context(), &[12]).unwrap();
        assert!(matches!(r.poll(), Ok(Event::NeedMaterial { pc: 0 })));
        assert!(matches!(
            r.feed_result(Err(Error::Resource)),
            Err(Error::Resource)
        ));
        assert!(r.is_consumed());
        assert!(matches!(r.poll(), Err(Error::Consumed)));
    }
    assert_eq!(budget.used_bytes(), p.resources().plan_reservation_bytes);
}

#[test]
fn replay_and_failed_peer_frame_burn_without_refunding_live_frame() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut b = builder(&[5], &budget);
    let x = b.input(0).unwrap();
    let y = b.multiply(x, x).unwrap();
    let z = b.multiply(x, y).unwrap();
    let p = b.finish(&[z]).unwrap();
    let mut a = p.runtime(0, context(), &[3]).unwrap();
    let mut b = p.runtime(1, context(), &[4]).unwrap();
    let [ma, mb] = p.issue_step(0, context()).unwrap();
    let Event::Frame(fa) = a.feed(ma).unwrap() else {
        panic!()
    };
    let Event::Frame(fb) = b.feed(mb).unwrap() else {
        panic!()
    };
    assert!(matches!(
        a.advance(fb.as_bytes()),
        Ok(Event::NeedMaterial { pc: 1 })
    ));
    assert!(matches!(
        b.advance(fa.as_bytes()),
        Ok(Event::NeedMaterial { pc: 1 })
    ));
    let [ma, mb] = p.issue_step(1, context()).unwrap();
    let fresh_a = a.feed(ma).unwrap();
    let fresh_b = b.feed(mb).unwrap();
    assert!(matches!(a.advance(fb.as_bytes()), Err(Error::Context)));
    assert!(a.is_consumed());
    assert!(matches!(b.advance(&[0; 9]), Err(Error::Context)));
    assert!(b.is_consumed());
    drop((a, b, fresh_a, fresh_b));
    assert_eq!(
        budget.used_bytes(),
        p.resources().plan_reservation_bytes + fa.as_bytes().len() + fb.as_bytes().len()
    );
    let bytes = fa.as_bytes().len() + fb.as_bytes().len();
    drop(p);
    assert_eq!(budget.used_bytes(), bytes);
    drop((fa, fb));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn complete_envelope_preflights_exact_budget_one_byte_short_and_live_headroom() {
    fn build(budget: &Budget) -> Result<Plan<'_>, Error> {
        let mut b = Builder::new([7; 32], b"exact-admission", &[5], budget)?;
        let x = b.input(0)?;
        let y = b.multiply(x, x)?;
        b.finish(&[y])
    }
    let large = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let p = build(&large).unwrap();
    let exact = p.resources().combined_reservation_bytes;
    drop(p);
    assert_eq!(large.used_bytes(), 0);
    let budget = Budget::new(exact).unwrap();
    let p = build(&budget).unwrap();
    let output = execute(&p, [&[2], &[3]]).unwrap();
    assert_eq!(reconstruct(&p, &output), [25]);
    drop((p, output));
    assert_eq!(budget.used_bytes(), 0);
    let short = Budget::new(exact - 1).unwrap();
    assert!(matches!(build(&short), Err(Error::Resource)));
    assert_eq!(short.used_bytes(), 0);
    let occupied = budget.reserve(1).unwrap();
    assert!(matches!(build(&budget), Err(Error::Resource)));
    assert_eq!(budget.used_bytes(), 1);
    drop(occupied);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn exact_envelope_prices_previous_frames_while_issuing_next_primitive() {
    fn build(budget: &Budget) -> Plan<'_> {
        let mut b = builder(&[5], budget);
        let x = b.input(0).unwrap();
        let square = b.multiply(x, x).unwrap();
        let cube = b.multiply(square, x).unwrap();
        b.finish(&[cube]).unwrap()
    }
    let large = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let p = build(&large);
    let exact = p.resources().combined_reservation_bytes;
    drop(p);
    let budget = Budget::new(exact).unwrap();
    let p = build(&budget);
    let mut a = p.runtime(0, context(), &[3]).unwrap();
    let mut b = p.runtime(1, context(), &[4]).unwrap();
    let [ma, mb] = p.issue_step(0, context()).unwrap();
    let Event::Frame(old_a) = a.feed(ma).unwrap() else {
        panic!()
    };
    let Event::Frame(old_b) = b.feed(mb).unwrap() else {
        panic!()
    };
    assert!(matches!(
        a.advance(old_b.as_bytes()),
        Ok(Event::NeedMaterial { pc: 1 })
    ));
    assert!(matches!(
        b.advance(old_a.as_bytes()),
        Ok(Event::NeedMaterial { pc: 1 })
    ));
    // Both previous exports remain live throughout fresh dealer issuance.
    let [ma, mb] = p.issue_step(1, context()).unwrap();
    let Event::Frame(next_a) = a.feed(ma).unwrap() else {
        panic!()
    };
    let Event::Frame(next_b) = b.feed(mb).unwrap() else {
        panic!()
    };
    let Event::Output(oa) = a.advance(next_b.as_bytes()).unwrap() else {
        panic!()
    };
    let Event::Output(ob) = b.advance(next_a.as_bytes()).unwrap() else {
        panic!()
    };
    let out = [oa, ob];
    assert_eq!(reconstruct(&p, &out), [23]);
    drop((a, b, next_a, next_b));
    let old_bytes = old_a.as_bytes().len() + old_b.as_bytes().len();
    drop(p);
    assert_eq!(budget.used_bytes(), old_bytes + 16);
    drop((old_a, old_b));
    assert_eq!(budget.used_bytes(), 16);
    drop(out);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn native_receipt_conserves_private_material_keys_triples_and_actual_frames() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut b = builder(&[5], &budget);
    let x = b.input(0).unwrap();
    let bit = b.lt_signed(x, 0).unwrap();
    let zero = b.constant(5, 0).unwrap();
    let selected = b.select(bit, zero, x).unwrap();
    let narrowed = b
        .convert(
            selected,
            conversions::Operation::Tr {
                shift: 1,
                range: conversions::SignedRange::full_ring(5).unwrap(),
            },
            4,
        )
        .unwrap();
    let extended = b
        .convert(narrowed, conversions::Operation::ZeroExtend, 5)
        .unwrap();
    let table = b.table(5, &[2, 3, 5, 7, 11, 13]).unwrap();
    let looked = b.lookup(extended, table).unwrap();
    let product = b.multiply(looked, looked).unwrap();
    let p = b.finish(&[product]).unwrap();
    let (outputs, frames, bytes) = execute_counted(&p, [&[18], &[21]]).unwrap();
    let receipt = p.resources();
    assert_eq!(receipt.frames_total, frames);
    assert_eq!(receipt.frame_bytes_total, bytes);
    let (mut private, mut keys, mut evaluations, mut triples) = (0, 0, 0, 0);
    for (pc, &instruction) in p.code.iter().enumerate() {
        match instruction {
            Instruction::Predicate(value, signed, threshold) => {
                let d = Descriptor::simple_comparison(p.registers[value].bits, signed, threshold)
                    .unwrap();
                let r = predicates::resources(&d).unwrap();
                // Count actual private Bit immediates through dealer handoff,
                // independently of the schedule's structural count formula.
                let compiled = d.compile(7).unwrap();
                let bit_words = compiled
                    .with_dealer_handoff(|h| {
                        Ok(h.boolean_program
                            .iter()
                            .filter(|i| i.opcode() == crate::fusefss::Opcode::Bit)
                            .count())
                    })
                    .unwrap();
                let key = crate::compact_dcf::RingComparison::resources(d.word().bits, 1).unwrap();
                private += 2
                    * (r.dcf_keys_per_party * key.encoded_key_bytes
                        + 8
                        + 8 * bit_words
                        + r.triples_per_party * crate::shared_wide_product::ring::TRIPLE_BYTES);
                keys += 2 * r.dcf_keys_per_party;
                evaluations += 2 * r.dcf_evaluations_per_party;
                triples += 2 * r.triples_per_party;
            }
            Instruction::Convert(..) | Instruction::Multiply(..) | Instruction::Select(..) => {
                let r = conversions::resources(p.conversion_spec(pc).unwrap()).unwrap();
                private += r.private_material_bytes_total;
                keys += 2 * r.dcf_keys_per_party;
                evaluations += 2 * r.dcf_evaluations_per_party;
                triples += 2 * r.triples;
            }
            Instruction::Lookup(_, table) => {
                let r = p.tables[table].resources(1).unwrap();
                private += 2 * r.private_key_bytes_per_party;
                keys += 2 * r.dcf_keys_per_party;
                evaluations += 2 * r.dcf_evaluations_per_party;
            }
            _ => {}
        }
    }
    assert_eq!(receipt.private_material_bytes_total, private);
    assert_eq!(receipt.dcf_keys_total, keys);
    assert_eq!(receipt.dcf_evaluations_total, evaluations);
    assert_eq!(receipt.triples_total, triples);
    assert_eq!(receipt.frames_total, 2 * receipt.frames_per_party);
    assert_eq!(receipt.frame_bytes_total, 2 * receipt.frame_bytes_per_party);
    assert!(receipt.public_table_reservation_bytes > 0);
    drop((p, outputs));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn fallible_allocation_paths_and_failed_feed_burn() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    budget.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(
        Builder::new([7; 32], b"contract", &[5], &budget),
        Err(Error::Allocation)
    ));
    assert_eq!(budget.used_bytes(), 0);
    budget.fail_after.store(-1, Ordering::SeqCst);
    let mut b = builder(&[5], &budget);
    let x = b.input(0).unwrap();
    let y = b.multiply(x, x).unwrap();
    let p = b.finish(&[y]).unwrap();
    let base = budget.used_bytes();
    for fail in 0..24 {
        budget.fail_after.store(fail, Ordering::SeqCst);
        let output = execute(&p, [&[2], &[3]]);
        if let Err(error) = &output {
            assert_eq!(*error, Error::Allocation);
        }
        drop(output);
        assert_eq!(budget.used_bytes(), base);
    }
    budget.fail_after.store(-1, Ordering::SeqCst);
    let mut r = p.runtime(0, context(), &[1]).unwrap();
    let [material, other] = p.issue_step(0, context()).unwrap();
    drop(other);
    budget.fail_after.store(0, Ordering::SeqCst);
    assert!(matches!(r.feed(material), Err(Error::Allocation)));
    assert!(r.is_consumed());
    drop(r);
    assert_eq!(budget.used_bytes(), base);
}

#[test]
fn allocation_failures_across_predicate_b2a_lookup_and_exports_refund() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut b = builder(&[4], &budget);
    let x = b.input(0).unwrap();
    let bit = b.lt_signed(x, 0).unwrap();
    let zero = b.constant(4, 0).unwrap();
    let index = b.select(bit, zero, x).unwrap();
    let table = b.table(4, &[2, 3, 5, 7]).unwrap();
    let value = b.lookup(index, table).unwrap();
    let p = b.finish(&[value]).unwrap();
    let base = budget.used_bytes();
    let mut succeeded = false;
    for fail in 0..100 {
        budget.fail_after.store(fail, Ordering::SeqCst);
        let output = execute(&p, [&[11], &[8]]);
        if let Err(error) = &output {
            assert_eq!(*error, Error::Allocation);
        }
        succeeded |= output.is_ok();
        drop(output);
        assert_eq!(budget.used_bytes(), base);
    }
    assert!(succeeded);
    budget.fail_after.store(-1, Ordering::SeqCst);
    drop(p);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn public_admission_digest_and_schedule_caps() {
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    assert!(matches!(
        Builder::new([1; 32], b"x", &[51], &budget),
        Err(Error::Domain)
    ));
    let mut b = builder(&[50], &budget);
    let x = b.input(0).unwrap();
    assert_eq!(b.lt_unsigned(x, (1u64 << 50) + 1), Err(Error::Domain));
    assert_eq!(b.lt_signed(x, 1i64 << 49), Err(Error::Domain));
    let p = b.finish(&[x]).unwrap();
    assert!(p.runtime(0, context(), &[1 << 50]).is_err());
    assert!(p.runtime(2, context(), &[0]).is_err());
    let mut b = builder(&[50], &budget);
    let x = b.input(0).unwrap();
    let y = b.narrow(x, 49).unwrap();
    let q = b.finish(&[y]).unwrap();
    assert_ne!(p.digest(), q.digest());
    drop((p, q));
    let mut b = builder(&[], &budget);
    for _ in 0..MAX_INSTRUCTIONS {
        b.constant(1, 0).unwrap();
    }
    assert_eq!(b.constant(1, 0), Err(Error::Resource));
    let p = b.finish(&[0]).unwrap();
    assert_eq!(p.resources().instructions, MAX_INSTRUCTIONS);
    assert!(p.resources().combined_reservation_bytes <= MAX_ALLOCATION_BYTES);
    drop(p);
    assert_eq!(budget.used_bytes(), 0);
}
