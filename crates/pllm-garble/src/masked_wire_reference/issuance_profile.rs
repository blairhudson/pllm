//! Opt-in native microprofile. Only counters and public-fixture digests escape.
//! This module is absent from installed builds; it does not issue reusable keys.

use super::*;
use serde_json::{json, Value};
use std::{collections::BTreeMap, time::Instant};

#[derive(Default)]
struct Cost {
    calls: usize,
    issue_ns: u64,
    encode_ns: u64,
    retire_ns: u64,
    party_bits: [usize; 2],
}

fn nanos(start: Instant) -> u64 {
    start.elapsed().as_nanos().try_into().unwrap()
}

fn component<T>(
    costs: &mut BTreeMap<&'static str, Cost>,
    name: &'static str,
    bodies: &mut [Bits; 2],
    issue: impl FnOnce() -> Result<[T; 2], Error>,
    encode: impl Fn(&T, &mut Bits),
) {
    let cost = costs.entry(name).or_default();
    let start = Instant::now();
    let keys = issue().unwrap();
    cost.issue_ns += nanos(start);
    let start = Instant::now();
    for (party, key) in keys.iter().enumerate() {
        let before = bodies[party].len;
        encode(key, &mut bodies[party]);
        cost.party_bits[party] += bodies[party].len - before;
    }
    cost.encode_ns += nanos(start);
    let start = Instant::now();
    drop(keys);
    cost.retire_ns += nanos(start);
    cost.calls += 1;
}

fn microprofile(lanes: usize, layout: Layout) -> Value {
    let resource = resources_with_layout(lanes, layout).unwrap();
    let mut costs = BTreeMap::<&str, Cost>::new();
    let mut bodies = [Bits::new(), Bits::new()];
    let start = Instant::now();
    for _ in 0..lanes {
        let before = Instant::now();
        let r = Zeroizing::new([
            random(N).unwrap(),
            random(N).unwrap(),
            random(N).unwrap(),
            random(N).unwrap(),
            random(Y).unwrap(),
            random(1).unwrap(),
            random(1).unwrap(),
            random(10).unwrap(),
            random(N).unwrap(),
            random(N).unwrap(),
            random(N).unwrap(),
            random(N).unwrap(),
        ]);
        let elapsed = nanos(before);
        let masks = costs.entry("wire_masks").or_default();
        masks.issue_ns += elapsed;
        masks.calls += 1;
        component(
            &mut costs,
            "gate_gapars",
            &mut bodies,
            || Shift::issue(N, 12, true, r[0], r[2]),
            Shift::encode,
        );
        component(
            &mut costs,
            "up_gapars",
            &mut bodies,
            || Shift::issue(N, 12, true, r[1], r[3]),
            Shift::encode,
        );
        let (bits, shift) = layout.reduce_bits();
        let input_mask = if layout == Layout::Fused { r[0] } else { r[2] };
        component(
            &mut costs,
            "index_reduce",
            &mut bodies,
            || Shift::issue(bits, shift, false, input_mask & mask(bits), r[4]),
            Shift::encode,
        );
        component(
            &mut costs,
            "sign_interval",
            &mut bodies,
            || {
                if layout == Layout::Fused {
                    Signs::issue_scaled(
                        INPUT_GATE_BITS,
                        18,
                        r[0] & mask(INPUT_GATE_BITS),
                        r[5],
                        r[6],
                    )
                } else {
                    Signs::issue(r[4], r[5], r[6])
                }
            },
            Signs::encode,
        );
        component(
            &mut costs,
            "clipped_index",
            &mut bodies,
            || {
                Select::issue(
                    10,
                    ((r[6] << 1) | r[5]) as usize,
                    r[4] & 1023,
                    r[7],
                    &[[0, 1023], [0, 1023], [-1, 0], [1, 0]],
                )
            },
            Select::encode,
        );
        component(
            &mut costs,
            "relu",
            &mut bodies,
            || Select::issue(N, r[5] as usize, r[2], r[8], &[[0, 0], [1, 0]]),
            Select::encode,
        );
        component(
            &mut costs,
            "residual_lookup",
            &mut bodies,
            || Lookup::issue(r[7], r[9]),
            Lookup::encode,
        );
        component(
            &mut costs,
            "product",
            &mut bodies,
            || Multiply::issue(r[8].wrapping_sub(r[9]) & mask(N), r[3], r[10]),
            Multiply::encode,
        );
        component(
            &mut costs,
            "output_gapars",
            &mut bodies,
            || Shift::issue(N, 12, true, r[10], r[11]),
            Shift::encode,
        );
    }
    let wall_ns = nanos(start);
    for body in &bodies {
        assert_eq!(body.len, resource.party_key_bits);
        assert_eq!(body.data.len(), resource.party_key_bytes);
    }
    let components: BTreeMap<_, _> = costs
        .into_iter()
        .map(|(name, cost)| {
            (
                name,
                json!({
                    "calls": cost.calls, "issue_ns": cost.issue_ns, "encode_ns": cost.encode_ns,
                    "retire_ns": cost.retire_ns, "party_key_bits": cost.party_bits,
                }),
            )
        })
        .collect();
    json!({"wall_ns": wall_ns, "components": components,
        "party_key_bits": resource.party_key_bits,
        "party_key_bytes": resource.party_key_bytes,
        "combined_allocation_estimate_bytes": resource.combined_allocation_estimate_bytes})
}

fn complete_profile(lanes: usize, layout: Layout) -> Value {
    let gate_values = [
        -(48 << 24),
        -(16 << 24),
        -(1 << 23),
        -1,
        0,
        1,
        1 << 24,
        16 << 24,
        48 << 24,
    ];
    let up_values = [128 << 24, -4097, -1, 0, 1, 4097, -(128 << 24)];
    let gate: Vec<i64> = (0..lanes)
        .map(|i| gate_values[i % gate_values.len()])
        .collect();
    let up: Vec<i64> = (0..lanes).map(|i| up_values[i % up_values.len()]).collect();
    let profile = Profile::new();
    let context = Context {
        plan: [1; 32],
        session: [2; 32],
        operation: profile.digest(),
        first_tensor_index: 0,
    };
    let resource = resources_with_layout(lanes, layout).unwrap();
    let total = Instant::now();
    let start = Instant::now();
    let (mut client, parties) = issue_reference_with_layout(context, lanes, layout).unwrap();
    let issuance_ns = nanos(start);
    let start = Instant::now();
    let [a, b] = parties;
    let packets = [
        transport::export_party(a).unwrap(),
        transport::export_party(b).unwrap(),
    ];
    let export_ns = nanos(start);
    let start = Instant::now();
    let mut a = transport::import_party(&packets[0], context, lanes, layout, 0).unwrap();
    let mut b = transport::import_party(&packets[1], context, lanes, layout, 1).unwrap();
    let import_ns = nanos(start);
    for packet in &packets {
        assert_eq!(
            packet.len(),
            transport::PACKET_HEADER + resource.party_key_bytes
        );
    }
    let start = Instant::now();
    let input = transport::encode_input(&mut client, &gate, &up).unwrap();
    let client_input_ns = nanos(start);
    let start = Instant::now();
    let mut left = transport::start(&mut a, &input).unwrap();
    let mut right = transport::start(&mut b, &input).unwrap();
    let worker_start_ns = nanos(start);
    let mut advance_ns = Vec::new();
    let mut phase_bytes = vec![[left.len(), right.len()]];
    for round in 0..layout.rounds() {
        let start = Instant::now();
        let result = (a.advance(&right).unwrap(), b.advance(&left).unwrap());
        advance_ns.push(nanos(start));
        match result {
            (Step::Send(l), Step::Send(r)) if round + 1 < layout.rounds() => {
                phase_bytes.push([l.len(), r.len()]);
                left = l;
                right = r;
            }
            (Step::Done(l), Step::Done(r)) if round + 1 == layout.rounds() => {
                let start = Instant::now();
                let l = transport::encode_output(l);
                let r = transport::encode_output(r);
                let outputs = transport::decode_output(client, &l, &r).unwrap();
                let output_codec_decode_ns = nanos(start);
                let wall_ns = nanos(total);
                assert_eq!(outputs.len(), lanes);
                let expected: Vec<_> = gate
                    .iter()
                    .zip(&up)
                    .map(|(&g, &u)| profile.evaluate(g, u).unwrap())
                    .collect();
                assert_eq!(outputs, expected);
                assert_eq!(
                    phase_bytes.iter().flatten().sum::<usize>(),
                    resource.peer_frame_bytes
                );
                let mut hash = Sha256::new();
                for value in outputs {
                    hash.update(value.to_le_bytes());
                }
                return json!({
                    "wall_ns": wall_ns, "issuance_ns": issuance_ns, "export_ns": export_ns,
                    "import_ns": import_ns, "client_input_ns": client_input_ns,
                    "output_codec_decode_ns": output_codec_decode_ns, "worker_start_ns": worker_start_ns,
                    "advance_ns": advance_ns, "phase_frame_bytes": phase_bytes,
                    "party_packet_bytes": [packets[0].len(), packets[1].len()],
                    "output_sha256": format!("{:x}", hash.finalize()), "checked_outputs": lanes,
                });
            }
            _ => panic!("unexpected masked-wire phase"),
        }
    }
    panic!("masked-wire profile did not complete")
}

#[test]
#[ignore = "opt-in release-mode issuance and codec measurement"]
fn profile_issuance_and_codecs() {
    assert!(!cfg!(debug_assertions), "profile requires --release");
    let mut cases = Vec::new();
    for lanes in [1, 64, 256] {
        for sample in 0..3 {
            for layout in if sample % 2 == 0 {
                [Layout::Unfused, Layout::Fused]
            } else {
                [Layout::Fused, Layout::Unfused]
            } {
                let micro = microprofile(lanes, layout);
                let complete = complete_profile(lanes, layout);
                cases.push(
                    json!({"layout": format!("{layout:?}").to_lowercase(), "lanes": lanes,
                    "sample": sample, "microprofile": micro, "complete": complete}),
                );
            }
        }
    }
    println!(
        "PLLM_ISSUANCE_PROFILE={}",
        json!({
            "schema": "pllm.masked_wire_issuance_profile.v1", "cases": cases,
            "timing_scope": "local wall-clock nanoseconds; per-primitive microprofile includes instrumentation; complete samples use ordinary one-use issuance, codecs and evaluation",
            "material_scope": "fresh primitive keys dropped after counting; separate complete-block samples consume fresh keys; no secret bytes retained",
            "memory_scope": "bounded at 256 lanes; no measured peak RSS or provider transport",
        })
    );
}
