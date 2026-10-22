//! Direct expanded-source-contract versus registry-constructor execution parity.
//! Both routes run the same semantic Plan, not a full expanded decoder oracle.
//! This does not assert complete-request D26 admission or new proof authority.
use super::*;

type Frame = (puma::Party, puma::Party, Vec<u8>);
type Components = Vec<Vec<(u64, u64)>>;

#[derive(Debug, PartialEq, Eq)]
struct Trace {
    // Complete frames in delivery order: no header stripping or sorting.
    wire: Vec<Frame>,
    // BOTH replicated components, grouped by party, output, then lane.
    shares: [Components; 3],
    // Includes public_blocks, private_blocks, pair_blocks AND aes_work.
    cursors: [puma::prss::Cursors; 3],
    advances: u64,
}

fn expanded_contract(manifest: &Manifest) -> Program {
    Program::compile_with_norms(
        &manifest.config,
        manifest.rows,
        manifest.outputs,
        &manifest.tensors,
        manifest.weights,
        manifest.execution,
        &manifest.norms,
    )
    .unwrap()
}

// Deterministic, nontrivial local splitting, matching the ordinary body harness.
// Only input components are reproduced; live runtime state is never copied.
fn input(party: puma::Party, clear: &[i64]) -> Secret {
    let mut value = Secret::new(clear.len()).unwrap();
    for (i, &word) in clear.iter().enumerate() {
        let a = 0xc751f00d12345678u64.wrapping_add(i as u64 * 17);
        let b = 0x254277811928f93du64.wrapping_add(i as u64 * 311);
        let c = (word as u64).wrapping_sub(a).wrapping_sub(b);
        let shares = [a, b, c];
        value.first[i] = shares[party as usize];
        value.second[i] = shares[(party as usize + 1) % 3];
    }
    value
}

fn owners(
    plan: &Arc<layer::Plan>,
    manifest: &Manifest,
    public: &Arc<PublicSource>,
    clear: &[Vec<i64>],
    registry_route: bool,
) -> [layer::Executor; 3] {
    let registry = SourceRegistry::from_manifest(manifest).unwrap();
    // Registry route deliberately never constructs an expanded Program.
    let program = (!registry_route).then(|| expanded_contract(manifest));
    semantic_runtimes(plan).map(|runtime| {
        assert_eq!(runtime.cursors(), puma::prss::Cursors::default());
        let inputs = clear
            .iter()
            .map(|words| input(runtime.party(), words))
            .collect();
        if registry_route {
            layer::Executor::new_with_registry(
                plan.clone(),
                runtime,
                inputs,
                Some(public.clone()),
                &registry,
            )
        } else {
            layer::Executor::new(
                plan.clone(),
                runtime,
                inputs,
                Some(public.clone()),
                program.as_ref(),
            )
        }
        .unwrap()
    })
    // Neither borrowed source contract survives construction.
}

fn trace(mut owners: [layer::Executor; 3], plan: &layer::Plan) -> Trace {
    let parties = [puma::Party::P0, puma::Party::P1, puma::Party::P2];
    let mut queue = Vec::<Frame>::new();
    let mut wire = Vec::new();
    let mut advances = 0;
    for _ in 0..500_000 {
        if queue.is_empty() {
            if owners.iter().all(layer::Executor::complete) {
                assert_eq!(advances, plan.coordinator_advances());
                let finished = owners.map(|owner| owner.finish().unwrap());
                let cursors = finished.each_ref().map(|(runtime, _)| {
                    assert_eq!(runtime.status(), multiply::Status::Ready);
                    let actual = runtime.cursors();
                    let budget = plan.work_budget().cursors();
                    assert_eq!(actual.public_blocks, budget.public_blocks);
                    assert_eq!(actual.pair_blocks, budget.pair_blocks);
                    assert!(actual.private_blocks <= budget.private_blocks);
                    assert!(actual.aes_work <= budget.aes_work);
                    actual
                });
                let shares = finished.each_ref().map(|(_, values)| {
                    (0..plan.output_count())
                        .map(|output| {
                            (0..values.output_len(output).unwrap())
                                .map(|i| values.word(output, i).unwrap())
                                .collect()
                        })
                        .collect()
                });
                return Trace {
                    wire,
                    shares,
                    cursors,
                    advances,
                };
            }
            advances += 1;
            for (party, owner) in parties.iter().zip(&mut owners) {
                assert!(!owner.complete(), "globally drained lockstep schedule");
                queue.extend(
                    owner
                        .advance()
                        .unwrap()
                        .into_iter()
                        .map(|(to, raw)| (*party, to, raw.as_bytes().to_vec())),
                );
            }
        } else {
            for (from, to, raw) in std::mem::take(&mut queue) {
                wire.push((from, to, raw.clone()));
                queue.extend(
                    owners[to as usize]
                        .receive(&raw, from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, raw)| (to, next, raw.as_bytes().to_vec())),
                );
            }
        }
        for owner in &owners {
            assert!(owner.live_bytes() <= plan.resources().peak_live_secret_bytes);
        }
    }
    panic!("registry constructor protocol did not finish");
}

fn reconstruct(trace: &Trace, output: usize) -> Vec<u64> {
    (0..trace.shares[0][output].len())
        .map(|lane| {
            for party in 0..3 {
                assert_eq!(
                    trace.shares[party][output][lane].1,
                    trace.shares[(party + 1) % 3][output][lane].0,
                );
            }
            trace
                .shares
                .iter()
                .fold(0u64, |sum, shares| sum.wrapping_add(shares[output][lane].0))
        })
        .collect()
}

fn guarded_plan(manifest: &Manifest) -> Arc<layer::Plan> {
    let registry = SourceRegistry::from_manifest(manifest).unwrap();
    let weight = Descriptor::find(&registry, "q_proj.weight");
    let bias = Descriptor::find(&registry, "q_proj.bias");
    assert_eq!(weight.bounds, Interval::new(ONE, ONE).unwrap());
    assert_eq!(bias.bounds, Interval::new(-ONE, -ONE).unwrap());
    let mut b = layer::Builder::new(
        WeightMode::Public,
        &[
            body::Register {
                lanes: 2,
                bounds: Interval::new(ONE, 2 * ONE).unwrap(),
            },
            body::Register {
                lanes: 129,
                bounds: Interval::new(-4 * ONE, 4 * ONE).unwrap(),
            },
        ],
    )
    .unwrap();
    let x = b.input(0).unwrap();
    let y = b.input(1).unwrap();
    let matrix = b
        .public_matrix(&x, weight.unchecked(), 1, 2, 2, false, true)
        .unwrap();
    let biased = b.bias(&matrix, bias.unchecked(), 2).unwrap();
    let (safe_matrix, first) = b.checked(&biased, ONE, None).unwrap();
    // Full tile + odd tail exercise Checked and AllValid shared handoffs.
    let (safe, sticky) = b.checked(&y, ONE, first.as_ref()).unwrap();
    // These lanes always pass at runtime but have a wider public envelope.
    // A later passing check must not clear either earlier rejection.
    let (later_safe, final_valid) = b
        .checked(&y.slice(0, 2).unwrap(), ONE, sticky.as_ref())
        .unwrap();
    // Noninteger coefficient exercises ordinary Op::Tensor shared handoffs,
    // including full/tail templates, with the SAME returned PRSS owner.
    let scaled = b.attention_scale(&safe, 2).unwrap();
    b.finish(&[
        biased,
        safe_matrix,
        first.unwrap(),
        safe,
        sticky.unwrap(),
        later_safe,
        final_valid.unwrap(),
        scaled,
    ])
    .unwrap()
}

#[test]
fn expanded_and_registry_constructors_match_exact_wire_both_shares_and_all_prss_cursors() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            let manifest = bias_manifest(); // Supported qwen2 adapter, not llama bias.
            let public = Arc::new(import(&manifest));
            let plan = guarded_plan(&manifest);
            assert!(plan.resources().admitted_bytes <= MAX_PROGRAM_BYTES);
            assert_eq!(plan.statistics().public_bias_vector_reads, 1);
            assert_eq!(plan.statistics().guard_tiles, 4);
            for failure in 0..3 {
                let matrix_fails = failure == 1;
                let tail_fails = failure == 2;
                let x = vec![if matrix_fails { 2 * ONE } else { ONE }, ONE];
                let mut y: Vec<_> = (0..129).map(|i| (i as i64 % 3 - 1) * (ONE / 2)).collect();
                if tail_fails {
                    y[128] = 2 * ONE;
                }
                let clear = [x, y.clone()];
                // Same PUBLIC seeds/context/input shares; completely separate
                // consuming owners and routers, never clone/reset protocol state.
                let expanded = owners(&plan, &manifest, &public, &clear, false);
                let registry = owners(&plan, &manifest, &public, &clear, true);
                assert_eq!(Arc::strong_count(&public), 7);
                let expected = trace(expanded, &plan);
                assert_eq!(Arc::strong_count(&public), 4);
                let actual = trace(registry, &plan);
                assert_eq!(Arc::strong_count(&public), 1);
                assert!(!actual.wire.is_empty());
                assert_eq!(actual, expected, "failure={failure}");
                assert!(actual.cursors.iter().all(|c| c.aes_work > 0));
                assert_eq!(
                    reconstruct(&actual, 0),
                    vec![(if matrix_fails { 2 * ONE } else { ONE }) as u64; 2]
                );
                assert_eq!(
                    reconstruct(&actual, 1),
                    vec![if matrix_fails { 0 } else { ONE as u64 }; 2]
                );
                assert_eq!(reconstruct(&actual, 2), vec![u64::from(!matrix_fails)]);
                if tail_fails {
                    y[128] = 0;
                }
                assert_eq!(
                    reconstruct(&actual, 3),
                    y.iter().map(|&v| v as u64).collect::<Vec<_>>()
                );
                assert_eq!(reconstruct(&actual, 4), vec![u64::from(failure == 0)]);
                assert_eq!(
                    reconstruct(&actual, 5),
                    y[..2].iter().map(|&v| v as u64).collect::<Vec<_>>()
                );
                assert_eq!(reconstruct(&actual, 6), vec![u64::from(failure == 0)]);
                // Check consistency of both components for the noninteger tensor
                // output too; exact stochastic result is compared above, not rounded
                // into a weaker cleartext-only tolerance assertion.
                reconstruct(&actual, 7);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn both_source_contract_routes_release_original_arc_on_every_lifecycle_exit() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            let manifest = bias_manifest();
            let public = Arc::new(import(&manifest));
            let registry = SourceRegistry::from_manifest(&manifest).unwrap();
            let descriptor = Descriptor::find(&registry, "q_proj.weight");
            let plan = single_source_plan(&descriptor, true);
            for registry_route in [false, true] {
                for exit in 0..5 {
                    let [mut owner, peer1, peer2] = owners(
                        &plan,
                        &manifest,
                        &public,
                        &[vec![ONE, 2 * ONE]],
                        registry_route,
                    );
                    drop((peer1, peer2));
                    assert_eq!(Arc::strong_count(&public), 2);
                    match exit {
                        0 => assert!(matches!(owner.finish(), Err(Error::Phase))),
                        1 => drop(owner),
                        2 | 3 => {
                            assert!(!owner.advance().unwrap().is_empty());
                            if exit == 2 {
                                drop(owner); // Pending tensor owns runtime, no leak.
                            } else {
                                owner.cancel();
                                owner.cancel();
                                assert_eq!(owner.live_bytes(), 0);
                                assert!(!owner.complete());
                                assert_eq!(Arc::strong_count(&public), 1);
                                assert!(matches!(owner.advance(), Err(Error::Retired)));
                                assert!(matches!(
                                    owner.receive(&[], puma::Party::P1),
                                    Err(Error::Retired)
                                ));
                                assert!(matches!(owner.finish(), Err(Error::Retired)));
                            }
                        }
                        4 => {
                            assert!(!owner.advance().unwrap().is_empty());
                            assert!(owner.receive(b"invalid", puma::Party::P1).is_err());
                            assert_eq!(owner.live_bytes(), 0);
                            assert_eq!(Arc::strong_count(&public), 1);
                            assert!(matches!(owner.advance(), Err(Error::Retired)));
                            assert!(matches!(owner.finish(), Err(Error::Retired)));
                        }
                        _ => unreachable!(),
                    }
                    assert_eq!(
                        Arc::strong_count(&public),
                        1,
                        "route={registry_route}, exit={exit}"
                    );
                    assert_eq!(Arc::strong_count(&plan), 1);
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn both_contracts_reject_descriptor_and_file_mismatch_before_bad_input_shape() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            let manifest = bias_manifest();
            let registry = SourceRegistry::from_manifest(&manifest).unwrap();
            let program = expanded_contract(&manifest);
            for matrix in [false, true] {
                let descriptor = Descriptor::find(
                    &registry,
                    if matrix {
                        "q_proj.weight"
                    } else {
                        "q_proj.bias"
                    },
                );
                for failure in 0..8 {
                    let mut descriptor = descriptor.clone();
                    let mut public = import(&manifest);
                    match failure {
                        0 => (), // Positive contract + invalid input -> Shape.
                        1 => descriptor.owner = usize::MAX,
                        2 => descriptor.key = "wrong.owner".into(),
                        3 => descriptor.digest = [9; 32],
                        4 => descriptor.lanes += 2,
                        5 => descriptor.bounds = Interval::new(-ONE, ONE).unwrap(),
                        6 => public.binding[0] ^= 1,
                        7 => {
                            public.files.remove(&descriptor.key);
                        }
                        _ => unreachable!(),
                    }
                    let public = Arc::new(public);
                    let plan = single_source_plan(&descriptor, matrix);
                    for registry_route in [false, true] {
                        let runtime = semantic_runtimes(&plan).into_iter().next().unwrap();
                        // Existing shape-validation seam. Input COUNT is correct,
                        // but lane count is wrong, so Phase cannot mask Binding.
                        let bad_input = vec![Secret::new(1).unwrap()];
                        let result = if registry_route {
                            layer::Executor::new_with_registry(
                                plan.clone(),
                                runtime,
                                bad_input,
                                Some(public.clone()),
                                &registry,
                            )
                        } else {
                            layer::Executor::new(
                                plan.clone(),
                                runtime,
                                bad_input,
                                Some(public.clone()),
                                Some(&program),
                            )
                        };
                        assert_eq!(
                            result.err(),
                            Some(if failure == 0 {
                                Error::Shape
                            } else {
                                Error::Binding
                            }),
                            "matrix={matrix}, failure={failure}, registry={registry_route}",
                        );
                        assert_eq!(Arc::strong_count(&public), 1);
                        assert_eq!(Arc::strong_count(&plan), 1);
                    }
                }
            }
            // This proves precedence over input-shape installation only. There
            // is no existing executor slot-allocation counter/fault seam here;
            // do not mislabel this as observed allocator-entry evidence.
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn checked_registry_keeps_original_source_certificate_binding_and_manifest_identity_obligations() {
    let mut manifest = manifest(false);
    let public = Arc::new(import(&manifest));
    let original_digest = manifest.tensors["model.embed_tokens.weight"].digest;
    // Legal metadata, identical file digest/extent, but not the certificate
    // recomputed by this original immutable source owner. Quoting must neither
    // reauthenticate it independently nor manufacture a replacement source.
    let tensor = manifest
        .tensors
        .get_mut("model.embed_tokens.weight")
        .unwrap();
    tensor.upper = 1;
    tensor.row_l1 = Some(1);
    tensor.row_squared_norm = Some(1);
    tensor.row_l1_squared_sum = Some(1);
    assert_eq!(tensor.digest, original_digest);
    let owners = SourceConstructionOwners {
        manifest: &manifest,
        incoming: None,
        public: Some(&public),
        retained_baseline_bytes: 0,
    };
    let quote = SourceRegistry::construction_quote_with_owners(&owners, MAX_PROGRAM_BYTES).unwrap();
    let validated = Program::validated_source_descriptors(&manifest).unwrap();
    let registry =
        SourceRegistry::from_validated_with_owners_checked(validated, &owners, quote.peak_bytes)
            .unwrap();
    assert!(matches!(
        public.check_registry(&registry),
        Err(Error::Binding)
    ));
    assert_eq!(Arc::strong_count(&public), 1);

    for field in 0..4 {
        let mut invalid = super::manifest(false);
        match field {
            0 => invalid.schema = "foreign.manifest.v1".into(),
            1 => invalid.source = [0; 32],
            2 => invalid.composition = [0; 32],
            3 => invalid.session = [0; 32],
            _ => unreachable!(),
        }
        // Source-only authority is deliberately not manifest identity/program
        // authentication. The ordinary Manifest::compile guard remains intact.
        SourceRegistry::from_manifest(&invalid).unwrap();
        assert_eq!(
            invalid.compile().err().as_deref(),
            Some("invalid replicated manifest identity")
        );
    }
}
