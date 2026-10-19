//! Synthetic native trie tests, compiled after the parent module hook is installed.
//! No private reconstruction occurs outside this synthetic test module.
use super::*;
use pllm_core::public_file::PublicFileWriter;
use puma::prss::{PublicContribution, SeedMessage};
use std::collections::VecDeque;

fn file(weights: &[u64]) -> PublicFile {
    let mut h = Sha256::new();
    for w in weights {
        h.update(w.to_le_bytes());
    }
    let mut writer =
        PublicFileWriter::new((weights.len() * 8) as u64, h.finalize().into()).unwrap();
    for chunk in weights.chunks(1024) {
        let bytes: Vec<_> = chunk.iter().flat_map(|w| w.to_le_bytes()).collect();
        writer.write(&bytes).unwrap();
    }
    writer.finish().unwrap()
}
fn plan(rows: usize, v: usize, width: usize, digest: [u8; 32]) -> Arc<Plan> {
    let proofs: Vec<_> = (0..rows)
        .map(|r| {
            if r % 2 == 0 {
                DomainOrigin::ValidatedInitialIds([11; 32])
            } else {
                DomainOrigin::NativeStableSelection([12; 32])
            }
        })
        .collect();
    Arc::new(
        Plan::new(
            rows,
            v,
            width,
            2,
            Interval::new(0, v as i64 - 1).unwrap(),
            &proofs,
            SourceBinding {
                owner: 1,
                source_commitment: [13; 32],
                file_digest: digest,
            },
        )
        .unwrap(),
    )
}
fn context(plan: &Plan) -> puma::Context {
    puma::Context::new([14; 32], plan.digest(), 0).unwrap()
}

fn runtimes(plan: &Plan) -> [PartyRuntime; 3] {
    let parties = [Party::P0, Party::P1, Party::P2];
    let mut setup = parties.map(|p| {
        multiply::SetupParty::new(context(plan), p, plan.work_budget().limits().unwrap()).unwrap()
    });
    let public = setup.each_mut().map(|s| s.contribution().unwrap().encode());
    for (i, s) in setup.iter_mut().enumerate() {
        for j in 0..3 {
            if i != j {
                s.receive_public(PublicContribution::decode(&public[j], parties[j]).unwrap())
                    .unwrap();
            }
        }
    }
    let mut seeds = setup
        .each_mut()
        .map(|s| Some(s.share_seed().unwrap().encode().unwrap()));
    for (i, s) in setup.iter_mut().enumerate() {
        let j = (i + 1) % 3;
        s.receive_seed(
            SeedMessage::decode(seeds[j].take().unwrap().as_bytes(), parties[j], parties[i])
                .unwrap(),
        )
        .unwrap();
    }
    setup.each_mut().map(|s| s.finish().unwrap())
}
fn shares(ids: &[u64]) -> [Secret; 3] {
    let a: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(r, _)| 0xffff_ffff_ffff_ffedu64.wrapping_sub((r as u64) << 18))
        .collect();
    let b: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(r, _)| 0x8000_0000_0003_ffffu64.wrapping_add((r as u64) << 36))
        .collect();
    let c: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(r, x)| x.wrapping_sub(a[r]).wrapping_sub(b[r]))
        .collect();
    let components = [a, b, c];
    std::array::from_fn(|i| {
        let mut out = Secret::new(ids.len()).unwrap();
        out.first.copy_from_slice(&components[i]);
        out.second.copy_from_slice(&components[(i + 1) % 3]);
        out
    })
}
fn owners(ids: &[u64], plan: Arc<Plan>, file: &PublicFile) -> [TrieLookup; 3] {
    let inputs = shares(ids);
    let ctx = context(&plan);
    let mut inputs = inputs.into_iter();
    runtimes(&plan)
        .map(|r| TrieLookup::new(r, inputs.next().unwrap(), plan.clone(), ctx, 2, 1, file).unwrap())
}
#[derive(Default, Debug)]
struct Frames {
    a2b: u64,
    b2a: u64,
    product: u64,
}
impl Frames {
    fn add(&mut self, class: Step, n: usize) {
        match class {
            Step::ReadyA2b => self.a2b += n as u64,
            Step::ReadyBits => self.b2a += n as u64,
            Step::ReadyProduct => self.product += n as u64,
            Step::ReadyStream => assert_eq!(n, 0),
            _ => panic!("not a ready phase: {class:?}"),
        }
    }
}
fn round(owners: &mut [TrieLookup; 3], file: &PublicFile, stats: &mut Frames) -> Result<(), Error> {
    let parties = [Party::P0, Party::P1, Party::P2];
    let class = owners[0].step;
    let mut queue: VecDeque<(Party, Party, PeerBytes)> = VecDeque::new();
    for (i, owner) in owners.iter_mut().enumerate() {
        assert_eq!(owner.step, class);
        let out = owner.start_step()?;
        stats.add(class, out.len());
        queue.extend(out.into_iter().map(|(to, f)| (parties[i], to, f)));
    }
    while let Some((from, to, frame)) = queue.pop_front() {
        assert_eq!(&frame.as_bytes()[..4], b"PUM1");
        let out = owners[to as usize].receive_encoded(frame.as_bytes(), from)?;
        stats.add(class, out.len());
        queue.extend(out.into_iter().map(|(dest, f)| (to, dest, f)));
    }
    for owner in owners {
        owner.finish_step(1, file)?;
    }
    Ok(())
}
fn to_selectors(owners: &mut [TrieLookup; 3], file: &PublicFile, stats: &mut Frames) {
    while owners[0].step != Step::ReadyStream {
        round(owners, file, stats).unwrap();
    }
}
fn assert_selectors(owners: &[TrieLookup; 3], ids: &[u64], v: usize) {
    for (r, &id) in ids.iter().enumerate() {
        let mut mass = 0_u64;
        for k in 0..v {
            let value = owners
                .iter()
                .fold(0_u64, |a, o| a.wrapping_add(o.parents[r].first[k]));
            assert_eq!(value, u64::from(k as u64 == id));
            mass = mass.wrapping_add(value);
            for i in 0..3 {
                assert_eq!(
                    owners[i].parents[r].second[k],
                    owners[(i + 1) % 3].parents[r].first[k]
                );
            }
        }
        assert_eq!(mass, 1);
    }
}
fn native_case(v: usize, ids: &[u64], width: usize) {
    let weights: Vec<_> = (0..v * width)
        .map(|k| {
            0x8000_0000_0000_0001u64.wrapping_add((k as u64).wrapping_mul(0xfedc_ba98_7654_3211))
        })
        .collect();
    let file = file(&weights);
    let plan = plan(ids.len(), v, width, file.digest());
    let mut owners = owners(ids, plan.clone(), &file);
    let mut stats = Frames::default();
    let usage = owners
        .each_ref()
        .map(|o| o.runtime.as_ref().unwrap().payload_usage());
    to_selectors(&mut owners, &file, &mut stats);
    assert_selectors(&owners, ids, v);
    // Independent complete local ring64 row-vector oracle for EACH component,
    // before the existing streaming helper sees the selectors.
    let expected: [[Vec<u64>; 2]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|component| {
            (0..ids.len() * width)
                .map(|j| {
                    let row = j / width;
                    let col = j % width;
                    let share = if component == 0 {
                        &owners[i].parents[row].first
                    } else {
                        &owners[i].parents[row].second
                    };
                    (0..v).fold(0_u64, |sum, k| {
                        sum.wrapping_add(share[k].wrapping_mul(weights[k * width + col]))
                    })
                })
                .collect()
        })
    });
    while !owners[0].complete() {
        round(&mut owners, &file, &mut stats).unwrap();
    }
    assert_eq!(stats.a2b, plan.logical.a2b_frames);
    assert_eq!(stats.b2a, plan.logical.b2a_frames);
    assert_eq!(stats.product, plan.logical.product_frames);
    for i in 0..3 {
        assert_eq!(
            owners[i].output.as_ref().unwrap().first.as_slice(),
            expected[i][0]
        );
        assert_eq!(
            owners[i].output.as_ref().unwrap().second.as_slice(),
            expected[i][1]
        );
        assert!(owners[i].peak_selectors <= plan.peak_selector_bytes());
    }
    let mut results = Vec::new();
    for owner in owners {
        let (mut runtime, result) = owner.finish().unwrap();
        let c = runtime.cursors();
        let b = plan.work_budget().cursors();
        assert_eq!(c.public_blocks, b.public_blocks);
        assert_eq!(c.pair_blocks, b.pair_blocks);
        assert_eq!(c.private_blocks, b.private_blocks);
        assert!(c.aes_work <= b.aes_work);
        // Exact ordinal limit is exhausted: even a one-lane local map is denied.
        let op = puma::Operation::new(
            runtime.context(),
            0,
            Party::P0,
            1,
            1,
            puma::Limits::default(),
        )
        .unwrap();
        let mut probe = puma::ReplicatedShare::from_local(
            op,
            runtime.party(),
            &[0],
            &[0],
            puma::Limits::default(),
        )
        .unwrap();
        assert!(runtime
            .affine(
                &mut probe,
                None,
                Affine {
                    left: 0,
                    right: 0,
                    constant: 0
                },
                multiply::InputBounds {
                    left: Interval::new(0, 0).unwrap(),
                    right: Interval::new(0, 0).unwrap()
                }
            )
            .is_err());
        runtime.cancel();
        results.push(result);
    }
    for (r, &id) in ids.iter().enumerate() {
        for col in 0..width {
            let x = results
                .iter()
                .fold(0_u64, |a, out| a.wrapping_add(out.first[r * width + col]));
            assert_eq!(x, weights[id as usize * width + col]);
        }
    }
    drop(results);
    for u in usage {
        assert_eq!(u.live_bytes(), 0);
        assert!(u.peak_bytes() <= u.limit_bytes());
    }
}

#[test]
fn trie_native_selector_and_both_component_oracles_default_stack() {
    // Explicitly no thread::Builder stack override / RUST_MIN_STACK workaround.
    for v in [2, 3, 5, 7, 16, 17, 31, 128, 129, 131] {
        native_case(
            v,
            &[0, v as u64 - 1, 1, v as u64 - 1, 0, 1, 0, v as u64 - 1],
            7,
        );
    }
    native_case(19, &[18, 0, 9], 3);
}

#[test]
fn trie_complete_eight_row_full_vocabulary_scalar_source_no_checkpoint() {
    native_case(151936, &[0, 131071, 131072, 151935, 1, 151935, 0, 17], 1);
}

#[test]
fn trie_public_plan_workload_budgets_and_overlap_are_exact() {
    for (rows, s, ordinals, pairs, aes, a2b, b2a, product_frames) in [
        (4, 2, 10707, 683981, 1367802, 42, 8, 32103),
        (64, 8, 641966, 41038412, 82067239, 1491, 540, 1924875),
    ] {
        let mut total = WorkBudget::default();
        let mut frames = [0_u64; 3];
        for n in rows..rows + s {
            for start in (0..n).step_by(8) {
                let p = plan(8.min(n - start), 151936, 896, [15; 32]);
                total = total.checked_add(p.work_budget()).unwrap();
                for (a, b) in frames.iter_mut().zip([
                    p.logical.a2b_frames,
                    p.logical.b2a_frames,
                    p.logical.product_frames,
                ]) {
                    *a += b;
                }
            }
        }
        assert_eq!(
            (
                total.operations(),
                total.cursors().pair_blocks,
                total.cursors().aes_work
            ),
            (ordinals, pairs, aes)
        );
        assert_eq!(frames, [a2b, b2a, product_frames]);
        total.limits().unwrap();
    }
    let p = plan(8, 151936, 896, [15; 32]);
    assert_eq!(p.geometry(), (8, 151936, 896));
    assert_eq!(p.logical_cost().secret_products, 8 * 151934);
    assert_eq!(p.logical.forced_component_copies, 2 * 8 * 7);
    assert_eq!(p.peak_selector_bytes(), 29171712);
    assert!(p.peak_extra_bytes().unwrap() < 30 << 20);
    assert!(p.vocabulary * p.rows > MAX_VALUES); // must remain row-sharded.
    assert!(p.vocabulary <= MAX_VALUES);
    eprintln!(
        "trie Plan={} Arc/deep={} owner={} peak={}",
        std::mem::size_of::<Plan>(),
        p.metadata_bytes(),
        std::mem::size_of::<TrieLookup>(),
        p.peak_extra_bytes().unwrap()
    );
    assert!(std::mem::size_of::<Plan>() < 4096);
    assert!(std::mem::size_of::<TrieLookup>() < 16384);
}

#[test]
fn trie_width18_128_plus16_b2a_and_arbitrary_ring_carries_default_stack() {
    // Generated 1.2 MiB scalar source, NOT a full matrix/checkpoint/model load.
    let file = file(&vec![0xfedc_ba98_7654_3210; 151936]);
    let ids = [0, 131071, 131072, 151935, 45, 1, 151935, 0];
    let p = plan(8, 151936, 1, file.digest());
    let mut owners = owners(&ids, p, &file);
    let mut stats = Frames::default();
    round(&mut owners, &file, &mut stats).unwrap(); // one A2B of all eight tokens.
    assert_eq!(stats.a2b, 21);
    round(&mut owners, &file, &mut stats).unwrap(); // first 128 bit lanes.
    assert_eq!(owners[0].bit_offset, 128);
    round(&mut owners, &file, &mut stats).unwrap(); // remaining 16 lanes.
    assert_eq!(stats.b2a, 8);
    assert!(owners
        .iter()
        .all(|o| o.packed.is_none() && o.bit_offset == 144));
    for (r, &id) in ids.iter().enumerate() {
        for bit in 0..18 {
            let x = owners.iter().fold(0_u64, |a, o| {
                a.wrapping_add(o.bits.as_ref().unwrap().first[r * 18 + bit])
            });
            assert_eq!(x, (id >> bit) & 1);
        }
    }
    for o in &owners {
        let c = o.runtime.as_ref().unwrap().cursors();
        assert_eq!((c.public_blocks, c.pair_blocks), (2, 240));
        assert!(c.aes_work <= 338);
    }
    for mut o in owners {
        let u = o.runtime.as_ref().unwrap().payload_usage();
        o.cancel();
        assert_eq!(u.live_bytes(), 0);
    }
}

#[test]
fn trie_domain_source_owner_and_profile_substitutions_are_rejected() {
    let f = file(&[1, 2, 3]);
    let source = SourceBinding {
        owner: 1,
        source_commitment: [13; 32],
        file_digest: f.digest(),
    };
    for bounds in [
        Interval::new(-1, 2).unwrap(),
        Interval::new(0, 3).unwrap(),
        Interval::new(1, 2).unwrap(),
    ] {
        assert!(Plan::new(
            1,
            3,
            1,
            2,
            bounds,
            &[DomainOrigin::ValidatedInitialIds([11; 32])],
            source
        )
        .is_err());
    }
    assert!(Plan::new(
        1,
        3,
        1,
        2,
        Interval::new(0, 2).unwrap(),
        &[DomainOrigin::ValidatedInitialIds([0; 32])],
        source
    )
    .is_err());
    assert!(Plan::new(
        1,
        1,
        1,
        2,
        Interval::new(0, 0).unwrap(),
        &[DomainOrigin::ValidatedInitialIds([11; 32])],
        source
    )
    .is_err());
    let p = plan(1, 3, 1, f.digest());
    let changed = Plan::new(
        1,
        3,
        1,
        2,
        Interval::new(0, 2).unwrap(),
        &[DomainOrigin::NativeStableSelection([11; 32])],
        source,
    )
    .unwrap();
    assert_ne!(p.digest(), changed.digest());
    for (input_owner, source_owner) in [(3, 1), (2, 2)] {
        let r = runtimes(&p).into_iter().next().unwrap();
        let t = shares(&[0]).into_iter().next().unwrap();
        let u = r.payload_usage();
        assert!(
            TrieLookup::new(r, t, p.clone(), context(&p), input_owner, source_owner, &f).is_err()
        );
        assert_eq!(u.live_bytes(), 0);
    }
    let r = runtimes(&p).into_iter().next().unwrap();
    let u = r.payload_usage();
    let t = shares(&[0]).into_iter().next().unwrap();
    let wrong_context = puma::Context::new([99; 32], p.digest(), 0).unwrap();
    assert!(TrieLookup::new(r, t, p.clone(), wrong_context, 2, 1, &f).is_err());
    assert_eq!(u.live_bytes(), 0);
    let wrong = file(&[9, 2, 3]);
    let mut o = owners(&[0], p.clone(), &f);
    to_selectors(&mut o, &f, &mut Frames::default());
    o[0].start_step().unwrap();
    assert!(o[0].finish_step(1, &wrong).is_err());
    assert!(o[0].retired());
}

fn advance_to(owners: &mut [TrieLookup; 3], f: &PublicFile, phase: Step) {
    while owners[0].step != phase {
        round(owners, f, &mut Frames::default()).unwrap();
    }
}
#[test]
fn trie_cancel_malformed_replay_and_early_barriers_burn_every_phase() {
    let f = file(&[1, 2, 3, 4, 5]);
    let p = plan(1, 5, 1, f.digest());
    for phase in [
        Step::ReadyA2b,
        Step::ReadyBits,
        Step::ReadyProduct,
        Step::ReadyStream,
    ] {
        for mode in 0..3 {
            let mut o = owners(&[4], p.clone(), &f);
            advance_to(&mut o, &f, phase);
            let usages = o
                .each_ref()
                .map(|x| x.runtime.as_ref().unwrap().payload_usage());
            if mode == 0 {
                for owner in &mut o {
                    drop(owner.start_step().unwrap());
                    owner.cancel();
                }
            } else if mode == 1 {
                drop(o[0].start_step().unwrap());
                assert!(o[0].receive_encoded(b"malformed", Party::P1).is_err());
            } else if phase != Step::ReadyStream {
                // An OT sender completes locally at issue; no local owner can
                // prove a GLOBAL barrier. Exercise an actually waiting role.
                if phase == Step::ReadyBits {
                    for owner in &mut o {
                        drop(owner.start_step().unwrap());
                    }
                    let waiting = o
                        .iter()
                        .position(|x| x.runtime.as_ref().unwrap().status() != Status::Complete)
                        .unwrap();
                    assert!(o[waiting].finish_step(1, &f).is_err());
                } else {
                    drop(o[0].start_step().unwrap());
                    assert!(o[0].finish_step(1, &f).is_err());
                }
            } else {
                assert!(o[0].finish_step(1, &f).is_err());
            }
            for owner in &mut o {
                owner.cancel();
                assert!(owner.retired());
                assert!(owner.start_step().is_err());
            }
            for usage in usages {
                assert_eq!(usage.live_bytes(), 0);
            }
        }
        if phase == Step::ReadyStream {
            continue;
        }
        let mut o = owners(&[4], p.clone(), &f);
        advance_to(&mut o, &f, phase);
        let mut queue = VecDeque::new();
        let parties = [Party::P0, Party::P1, Party::P2];
        for (i, owner) in o.iter_mut().enumerate() {
            queue.extend(
                owner
                    .start_step()
                    .unwrap()
                    .into_iter()
                    .map(|(to, frame)| (parties[i], to, frame)),
            );
        }
        let (from, to, frame) = queue.pop_front().unwrap();
        drop(
            o[to as usize]
                .receive_encoded(frame.as_bytes(), from)
                .unwrap(),
        );
        assert!(o[to as usize]
            .receive_encoded(frame.as_bytes(), from)
            .is_err());
        assert!(o[to as usize].retired());
        drop(queue);
        drop(frame);
        for owner in &mut o {
            owner.cancel();
        }
    }
}

#[test]
fn trie_owned_allocation_failures_retire_initial_and_each_later_phase() {
    let f = file(&[1, 2, 3, 4, 5]);
    let p = plan(1, 5, 1, f.digest());
    for remaining in 0..2 {
        let r = runtimes(&p).into_iter().next().unwrap();
        let u = r.payload_usage();
        let t = shares(&[4]).into_iter().next().unwrap();
        assert!(TrieLookup::construct(
            r,
            t,
            p.clone(),
            context(&p),
            2,
            1,
            &f,
            Allocations {
                remaining: Some(remaining),
                calls: 0
            }
        )
        .is_err());
        assert_eq!(u.live_bytes(), 0);
    }
    for phase in [
        Step::ReadyA2b,
        Step::ReadyBits,
        Step::ReadyProduct,
        Step::ReadyStream,
    ] {
        let mut o = owners(&[4], p.clone(), &f);
        advance_to(&mut o, &f, phase);
        let u = o[0].runtime.as_ref().unwrap().payload_usage();
        o[0].alloc.remaining = Some(0);
        assert!(o[0].start_step().is_err());
        assert!(o[0].retired());
        assert_eq!(u.live_bytes(), 0);
    }
    // Stream scratch is allocated in finish, after start's empty frame list.
    let mut o = owners(&[4], p, &f);
    advance_to(&mut o, &f, Step::ReadyStream);
    drop(o[0].start_step().unwrap());
    o[0].alloc.remaining = Some(0);
    let u = o[0].runtime.as_ref().unwrap().payload_usage();
    assert!(o[0].finish_step(1, &f).is_err());
    assert!(o[0].retired());
    assert_eq!(u.live_bytes(), 0);
}

#[test]
fn trie_exhaustive_owner_allocation_countdown_sweep() {
    let f = file(&[1, 2, 3, 4, 5]);
    let p = plan(1, 5, 1, f.digest());
    let mut failures = 0;
    let mut successes = 0;
    for stop in 0..96 {
        let mut o = owners(&[4], p.clone(), &f);
        let usage = o
            .each_ref()
            .map(|x| x.runtime.as_ref().unwrap().payload_usage());
        o[0].alloc.remaining = Some(stop);
        let result = (|| {
            while !o[0].complete() {
                round(&mut o, &f, &mut Frames::default())?;
            }
            Ok::<_, Error>(())
        })();
        if result.is_err() {
            failures += 1;
            assert!(o[0].retired());
        } else {
            successes += 1;
        }
        for owner in &mut o {
            owner.cancel();
        }
        for u in usage {
            assert_eq!(u.live_bytes(), 0);
        }
    }
    assert!(failures > 10 && successes > 10);
}

#[test]
fn trie_cancel_each_inflight_conversion_ot_and_product_frame_boundary() {
    let f = file(&[1, 2, 3, 4, 5]);
    let p = plan(1, 5, 1, f.digest());
    let parties = [Party::P0, Party::P1, Party::P2];
    for phase in [Step::ReadyA2b, Step::ReadyBits, Step::ReadyProduct] {
        for cut in 0..=24 {
            let mut o = owners(&[4], p.clone(), &f);
            advance_to(&mut o, &f, phase);
            let usages = o
                .each_ref()
                .map(|x| x.runtime.as_ref().unwrap().payload_usage());
            let mut queue = VecDeque::new();
            for (i, owner) in o.iter_mut().enumerate() {
                queue.extend(
                    owner
                        .start_step()
                        .unwrap()
                        .into_iter()
                        .map(|(to, frame)| (parties[i], to, frame)),
                );
            }
            for _ in 0..cut {
                let Some((from, to, frame)) = queue.pop_front() else {
                    break;
                };
                let replies = o[to as usize]
                    .receive_encoded(frame.as_bytes(), from)
                    .unwrap();
                queue.extend(replies.into_iter().map(|(dest, frame)| (to, dest, frame)));
            }
            // Frames intentionally retained across cancellation remain charged.
            for owner in &mut o {
                owner.cancel();
                assert!(owner.retired());
            }
            drop(queue);
            for u in usages {
                assert_eq!(u.live_bytes(), 0);
            }
        }
    }
}
