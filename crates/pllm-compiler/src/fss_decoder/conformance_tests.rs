use super::*;

#[test]
#[ignore = "fresh-material wavefront full-decoder regression; explicit focused run"]
fn wavefront_preserves_every_frozen_semantic_tensor_and_private_feedback() {
    frozen_semantic_tensors_with_schedule(OnlineSchedule::WavefrontV1);
}

#[test]
#[ignore = "fresh-material compact tensor full-decoder regression; explicit focused run"]
fn tensor_preserves_every_frozen_semantic_tensor_and_private_feedback() {
    frozen_semantic_tensors_with_schedule(OnlineSchedule::TensorV1);
}

fn frozen_semantic_tensors_with_schedule(online: OnlineSchedule) {
    let inputs = read_json(fixture_dir().join("inputs.json"));
    let graphs = read_json(fixture_dir().join("semantic-graphs.json"));
    let expected = read_json(fixture_dir().join("semantic-expected.json"));
    for (index, case) in inputs["cases"].as_array().unwrap().iter().enumerate() {
        for decoder in [false, true] {
            // Dense full-prefix fixtures use explicit Decoder capacity: the
            // additional schedule-construction work exceeds Capsule's ceiling.
            // Learned-position/input-output fixtures retain their Capsule path.
            if decoder == (case["profile"] == "fusefss.gpt2-complete-ring50-q12.v1") {
                continue;
            }
            let semantic = semantic(case);
            let resident = source(case, &semantic).unwrap();
            let source = if decoder {
                into_decoder_files(resident)
            } else {
                resident
            };
            let limits = Limits::for_capacity(source.capacity());
            let program = Program::compile_with_schedule(
                &semantic,
                source,
                number(case, "generation_count"),
                limits,
                online,
            )
            .unwrap_or_else(|e| panic!("{} decoder={decoder}: {e:?}", case["id"]));
            let (tokens, trace, actual) = execute(&program, case, index);
            let result =
                semantic_native_result(&semantic, case, &graphs["cases"][index], &tokens, &trace);
            assert_eq!(
                result, expected["results"][index],
                "{} decoder={decoder}",
                case["id"]
            );
            assert!(actual.frames > 0);
        }
    }
}

#[test]
#[ignore = "fresh-material wavefront D14 regression; explicit focused run"]
fn wavefront_checked_profile_preserves_frozen_generated_qwen() {
    frozen_generated_qwen_with_schedule(true, OnlineSchedule::WavefrontV1);
}

#[test]
#[ignore = "fresh-material compact tensor D14 regression; explicit focused run"]
fn tensor_checked_profile_preserves_frozen_generated_qwen() {
    frozen_generated_qwen_with_schedule(true, OnlineSchedule::TensorV1);
}

#[test]
#[ignore = "fresh-material shared-public-chunk full-decoder regression; explicit focused run"]
fn tensor_shared_lowered_chunks_preserve_frozen_generated_qwen() {
    // Includes complete row reductions, adjacent stable selection, fixed-T
    // private feedback and final acceptance. Native execution separately checks
    // every fresh issued pair, material byte and frame against admission.
    frozen_generated_qwen_with_schedule(false, OnlineSchedule::TensorV1);
}

/// Ordinary regression: every physical node and private feedback step is checked
/// against the independently frozen integer result. No export/evidence bundle.
#[test]
#[ignore = "fresh-material full generated Qwen regression; explicit focused run"]
fn decoder_row_energy_matches_frozen_generated_qwen() {
    frozen_generated_qwen_with_policy(false);
}

#[test]
#[ignore = "fresh-material D14 full generated Qwen regression; explicit focused run"]
fn checked_profile_matches_frozen_generated_qwen() {
    frozen_generated_qwen_with_policy(true);
}

fn frozen_generated_qwen_with_policy(checked: bool) {
    frozen_generated_qwen_with_schedule(checked, OnlineSchedule::Serial);
}

fn into_decoder_files(resident: Source) -> Source {
    let digest = resident.digest();
    let mut tensors = Vec::new();
    for tensor in resident.tensors {
        if tensor.shape().len() != 2 {
            tensors.push(tensor);
            continue;
        }
        let mut import = crate::fss_decoder::source::TensorImport::new(
            tensor.id(),
            tensor.shape(),
            Capacity::Decoder,
        )
        .unwrap();
        for values in tensor.values().unwrap().chunks(8192) {
            let mut frame = Vec::new();
            for &code in values {
                let word = (code as f32 / 4096.0).to_bits();
                assert_eq!(
                    crate::fss_decoder::source::decode_ieee(WeightDtype::F32, word).unwrap(),
                    code
                );
                frame.extend_from_slice(&word.to_le_bytes());
            }
            import.append(WeightDtype::F32, &frame).unwrap();
        }
        tensors.push(import.finish().unwrap());
    }
    let mut source = Source::new_with_capacity(tensors, Capacity::Decoder).unwrap();
    if !resident.rotary.is_empty() {
        source = source.with_rotary(resident.rotary).unwrap();
    }
    assert_eq!(source.digest(), digest);
    source
}

fn frozen_generated_qwen_with_schedule(checked: bool, online: OnlineSchedule) {
    let inputs = read_json(fixture_dir().join("inputs.json"));
    let graphs = read_json(fixture_dir().join("semantic-graphs.json"));
    let expected = read_json(fixture_dir().join("semantic-expected.json"));
    let case = &inputs["cases"][2];
    assert_eq!(case["id"], "qwen2-tied-generated");
    let semantic = semantic(case);
    let resident = source(case, &semantic).unwrap();
    let mut source = into_decoder_files(resident);
    if checked {
        source = source.with_checked_ranges();
    }
    let program = Program::compile_with_schedule(
        &semantic,
        source,
        number(case, "generation_count"),
        Limits::for_capacity(Capacity::Decoder),
        online,
    )
    .unwrap();
    let (tokens, trace, _) = execute(&program, case, 2);
    let actual = semantic_native_result(&semantic, case, &graphs["cases"][2], &tokens, &trace);
    assert_eq!(actual["nodes"].as_array().unwrap().len(), 186);
    assert_eq!(actual, expected["results"][2]);
}

// Preserve original witnessed rejection/source snapshots byte-for-byte. Future
// reruns of the original stricter schema get distinct artifact names.
fn original_rerun_json(name: &str, value: &Value) {
    write_json(&format!("original-exchange-rerun-v2-{name}"), value);
}

#[test]
fn frozen_exchange_admission() {
    let before = snapshot();
    original_rerun_json("before.json", &before);
    let input = read_json(fixture_dir().join("inputs.json"));
    let expected = read_json(fixture_dir().join("expected.json"));
    assert_eq!(input["schema"], "pllm.fusefss.decoder-inputs.v1");
    let mut cases = Vec::new();
    for (case, oracle) in input["cases"]
        .as_array()
        .unwrap()
        .iter()
        .zip(expected["results"].as_array().unwrap())
    {
        assert_eq!(case["id"], oracle["fixture_id"]);
        let semantic = semantic(case);
        let source =
            source(case, &semantic).expect("frozen source imports through native source decoder");
        let source_digest = hex(source.digest());
        let map: Vec<_> = semantic
            .prefill
            .operations
            .iter()
            .enumerate()
            .map(|(index, op)| {
                json!({"semantic_index":index,"id":op.id,"operator":format!("{:?}",op.operator),
                "inputs":op.inputs,"declared_shape":op.output_shape,"attributes":op.attributes})
            })
            .collect();
        let entry = match Program::compile(
            &semantic,
            source,
            number(case, "generation_count"),
            Limits::default(),
        ) {
            Ok(program) => {
                json!({"fixture_id":case["id"],"admitted":true,"program_digest":hex(program.digest()),
                "source_digest":source_digest,"resources":format!("{:?}",program.resources()),
                "semantic_map":map,"expected_tensors":oracle["tensors"].as_array().unwrap().len()})
            }
            Err(error) => {
                json!({"fixture_id":case["id"],"admitted":false,"error":format!("{error:?}"),
                "source_digest":source_digest,"semantic_map":map,"expected_tensors":oracle["tensors"].as_array().unwrap().len(),
                "issued_materials":0,"actual_frame_bytes":0,"native_replay":false})
            }
        };
        eprintln!(
            "admission {}: {} {}",
            case["id"], entry["admitted"], entry["error"]
        );
        cases.push(entry);
        original_rerun_json("admission.partial.json", &json!({"cases":cases}));
    }
    let after = snapshot();
    original_rerun_json("after.json", &after);
    original_rerun_json(
        "admission.json",
        &json!({"schema":"pllm.fusefss.decoder-native-admission.v1",
        "scope":"frozen-generated-only-test-module","cases":cases,
        "limits":format!("{:?}",Limits::default()),"frozen_fixture_bytes_unchanged":before==after}),
    );
    assert_eq!(before, after, "source/fixture files changed during run");
}

#[test]
fn frozen_rope_signed_product_order_matches_approved_oracle() {
    let before = snapshot();
    let inputs = read_json(fixture_dir().join("inputs.json"));
    let expected = read_json(fixture_dir().join("expected.json"));
    let c = &inputs["cases"][2];
    let result = &expected["results"][2];
    assert_eq!(c["family"], "qwen2");
    let tensor = |name: &str| -> Vec<i64> {
        result["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == name)
            .unwrap()["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect()
    };
    let q = tensor("step.0.layer.0.q.physical");
    let expected_output = tensor("step.0.layer.0.q_rope.output");
    let cos = public_tensor("cos", &c["tensors"]["rope.cos"]).unwrap();
    let sin = public_tensor("sin", &c["tensors"]["rope.sin"]).unwrap();
    let d = number(&c["dimensions"], "head_dim");
    let sequence = c["prefix"].as_array().unwrap().len();
    let h = number(&c["dimensions"], "heads");
    let mut witness = None;
    'outer: for head in 0..h {
        for position in 0..sequence {
            for feature in 0..d / 2 {
                let i = (head * sequence + position) * d + feature;
                let j = i + d / 2;
                if (q[j] * sin.values().unwrap()[position * d + feature]).rem_euclid(4096) != 0 {
                    witness = Some((head, position, feature, i, j));
                    break 'outer;
                }
            }
        }
    }
    let (head, position, feature, i, j) =
        witness.expect("fixture distinguishes negative-product versus subtraction floor");
    let budget = pllm_garble::fusefss_backend::Budget::new(
        pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES,
    )
    .unwrap();
    let binding = kernels::Binding {
        source_digest: [41; 32],
        contract_digest: [42; 32],
    };
    let lo = q[i].min(q[j]);
    let hi = q[i].max(q[j]);
    let cosine = cos.values().unwrap()[position * d + feature];
    let sine = sin.values().unwrap()[position * d + feature];
    let (plan, _) = kernels::rope(
        binding,
        [gates::InputRange { min: lo, max: hi }; 2],
        cosine,
        sine,
        &budget,
    )
    .unwrap();
    let context = Context {
        plan: plan.digest(),
        session: [43; 32],
        operation: [44; 32],
        first_tensor_index: 0,
    };
    let a = [1234567, 7654321];
    let b = [
        (q[i] as u64).wrapping_sub(a[0]) & MASK,
        (q[j] as u64).wrapping_sub(a[1]) & MASK,
    ];
    let mut actual = Actual::default();
    let [left, right] =
        native_pair(&plan, OnlineSchedule::Serial, context, &a, &b, &mut actual).unwrap();
    // Isolated native witness from frozen operands, NOT a complete native model.
    let native = reconstruct(&left.shares, &right.shares);
    let frozen = [expected_output[i], expected_output[j]];
    write_json(
        "rope-signed-product-correction-v2.json",
        &json!({"schema":"pllm.fusefss.decoder-rope-signed-product-replay.v1",
        "scope":"isolated-native-rope-kernel-on-frozen-oracle-operands-not-whole-model-replay",
        "fixture_id":c["id"],"tensor":"step.0.layer.0.q_rope.output","head":head,"position":position,"feature":feature,
        "flat_indices":[i,j],"input_q12":[q[i],q[j]],"cos_q12":cosine,"sin_q12":sine,
        "frozen_expected":frozen,"actual_native":native,"matches":native==frozen,
        "issued_fresh_material_pairs":actual.issued_pairs,"actual_frames":actual.frames,"actual_frame_bytes":actual.frame_bytes,
        "native_plan_digest":hex(plan.digest()),"before":before,"after":snapshot(),
        "fixture_modified":false,"expected_modified":false}),
    );
    assert_eq!(
        native, frozen,
        "D13 signed-product contract: negation must precede floor"
    );
    assert_eq!(
        native[0],
        (q[i] * cosine).div_euclid(4096) + ((-q[j]) * sine).div_euclid(4096)
    );
}

#[test]
#[ignore = "explicit fresh-material full decoder export; run independent checker separately"]
fn attempt_frozen_exchange_native_replay_without_fabricating_checkpoints() {
    let before = snapshot();
    let inputs = read_json(fixture_dir().join("inputs.json"));
    let expected = read_json(fixture_dir().join("expected.json"));
    let mut results = Vec::new();
    let mut receipts = Vec::new();
    for (case_index, (c, oracle)) in inputs["cases"]
        .as_array()
        .unwrap()
        .iter()
        .zip(expected["results"].as_array().unwrap())
        .enumerate()
    {
        let semantic = semantic(c);
        let src = source(c, &semantic).unwrap();
        let program = match Program::compile(
            &semantic,
            src,
            number(c, "generation_count"),
            Limits::default(),
        ) {
            Ok(program) => program,
            Err(error) => {
                receipts.push(json!({"fixture_id":c["id"],"admitted":false,"blocker":format!("Program::compile: {error:?}"),
                    "issued_fresh_material_pairs":0,"actual_frames":0,"actual_frame_bytes":0,"native_checkpoints":0,
                    "full_exchange_complete":false}));
                continue;
            }
        };
        let (tokens, trace, actual) = execute(&program, c, case_index);
        let (mut checkpoints, map) = mapped_checkpoints(&semantic, c, &trace);
        // Native external quantization of public tensors is an observed native
        // checkpoint. Never use expected payload values to fill export holes.
        for (name, value) in c["tensors"].as_object().unwrap() {
            checkpoints.insert(
                format!("public.{name}"),
                public_tensor(name, value)
                    .unwrap()
                    .values()
                    .unwrap()
                    .to_vec(),
            );
        }
        checkpoints.insert(
            "public.attention_scale".into(),
            public_tensor("scale", &c["attention_scale"])
                .unwrap()
                .values()
                .unwrap()
                .to_vec(),
        );
        let mut tensors = Vec::new();
        let mut missing = Vec::new();
        let mut mismatches = Vec::new();
        for template in oracle["tensors"].as_array().unwrap() {
            let name = template["name"].as_str().unwrap();
            if let Some(values) = checkpoints.remove(name) {
                if json!(values) != template["values"] {
                    mismatches.push(
                        json!({"name":name,"native":values,"frozen_expected":template["values"]}),
                    );
                }
                let mut output = template.clone();
                output["values"] = json!(values);
                tensors.push(output);
            } else {
                missing.push(name);
            }
        }
        let mut prefix: Vec<_> = c["prefix"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let initial = prefix.len();
        let mut states = Vec::new();
        for (step, token) in tokens.iter().enumerate() {
            prefix.push(*token);
            states.push(json!({"step":step,"prefix_length_in":initial+step,"prefix_length_out":initial+step+1,
                "positions_recomputed":(0..initial+step).collect::<Vec<_>>(),"persistent_kv":false,
                "selected_id_private":token,"prefix_ids_private":prefix,"released":false}));
        }
        let mut canonical_input = serde_json::to_vec(c).unwrap();
        canonical_input.push(b'\n');
        results.push(json!({"schema":"pllm.fusefss.decoder-exchange.v1","profile":program.profile(),"graph_status":c["graph_status"],
            "fixture_id":c["id"],"input_sha256":hash(&canonical_input),"tensors":tensors,"states":states,
            "final":{"recipient":"client-only","token_ids":prefix,"generated_ids":tokens,"selection_count":number(c,"generation_count"),"persistent_kv":false}}));
        receipts.push(json!({"fixture_id":c["id"],"admitted":true,"program_digest":hex(program.digest()),
            "native_operator_observations":trace.len(),"exported_tensors":tensors.len(),"missing_checkpoint_names":missing,
            "mismatches":mismatches,"semantic_map":map,"issued_fresh_material_pairs":actual.issued_pairs,
            "actual_frames":actual.frames,"actual_frame_bytes":actual.frame_bytes,
            "issued_material_logical_bytes":actual.material_logical_bytes,"material_serialization_claimed":false,
            "declared_resources":format!("{:?}",program.resources()),"full_exchange_complete":missing.is_empty()&&mismatches.is_empty()}));
        eprintln!(
            "native {}: {} exported, {} unavailable, {} mismatches",
            c["id"],
            tensors.len(),
            missing.len(),
            mismatches.len()
        );
    }
    original_rerun_json(
        "native-output.json",
        &json!({"schema":"pllm.fusefss.decoder-batch.v1","results":results}),
    );
    let complete =
        receipts.len() == 5 && receipts.iter().all(|r| r["full_exchange_complete"] == true);
    original_rerun_json(
        "replay-receipt.json",
        &json!({"schema":"pllm.fusefss.decoder-native-replay-receipt.v1",
        "scope":"test-only-generated-checkpoints-no-production-disclosure","cases":receipts,"before":before,"after":snapshot(),
        "expected_weakened":false,"caps_changed":false,"full_native_exchange_complete":complete,
        "unavailable_checkpoint_policy":"omit-and-fail-strict-checker-never-copy-or-clear-recompute-expected"}),
    );
    assert_eq!(before, snapshot());
}

#[test]
#[ignore = "explicit fresh-material semantic decoder export; run independent checker separately"]
fn attempt_strict_semantic_execution_replay() {
    let before = snapshot();
    let inputs = read_json(fixture_dir().join("inputs.json"));
    let graphs = read_json(fixture_dir().join("semantic-graphs.json"));
    assert_eq!(graphs["schema"], "pllm.fusefss.semantic-graphs.v1");
    let cases = inputs["cases"].as_array().unwrap();
    let fixed = graphs["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5);
    assert_eq!(fixed.len(), cases.len());
    let mut results = Vec::new();
    let mut receipts = Vec::new();
    for (case_index, (c, graph)) in cases.iter().zip(fixed).enumerate() {
        let semantic = semantic(c);
        assert_eq!(graph["fixture_id"], c["id"]);
        assert_eq!(json!(semantic_operator_map(&semantic)), graph["operators"]);
        let source = source(c, &semantic).unwrap();
        let source_digest = hex(source.digest());
        let program = match Program::compile(
            &semantic,
            source,
            number(c, "generation_count"),
            Limits::default(),
        ) {
            Ok(program) => program,
            Err(error) => {
                receipts.push(json!({"fixture_id":c["id"],"admitted":false,
                    "source_digest":source_digest,"error":format!("{error:?}"),
                    "issued_fresh_material_pairs":0,"actual_frames":0,"actual_frame_bytes":0,
                    "native_node_observations":0,"full_semantic_result_produced":false}));
                continue;
            }
        };
        let (tokens, trace, actual) = execute(&program, c, case_index);
        let output = semantic_native_result(&semantic, c, graph, &tokens, &trace);
        receipts.push(json!({"fixture_id":c["id"],"admitted":true,"source_digest":source_digest,
            "program_digest":hex(program.digest()),"semantic_graph_sha256":canonical_digest(graph),
            "native_node_observations":trace.len(),"full_semantic_result_produced":true,
            "issued_fresh_material_pairs":actual.issued_pairs,"actual_frames":actual.frames,
            "actual_frame_bytes":actual.frame_bytes,"issued_material_logical_bytes":actual.material_logical_bytes,
            "material_serialization_claimed":false,"resources":format!("{:?}",program.resources())}));
        results.push(output);
    }
    write_json(
        "semantic-v1-native-output.json",
        &json!({"schema":"pllm.fusefss.semantic-execution-batch.v1","results":results}),
    );
    write_json(
        "semantic-v1-receipt.json",
        &json!({"schema":"pllm.fusefss.semantic-native-producer-receipt.v1",
        "scope":"test-only-generated-executed-semantic-nodes-not-original-5125-tensor-exchange",
        "cases":receipts,"before":before,"after":snapshot(),"caps_changed":false,
        "full_semantic_batch_produced":results.len()==cases.len(),"independent_python_check_required":true,
        "original_exchange_pass_claimed":false,"clear_expected_values_used_in_native_export":false}),
    );
    assert_eq!(
        before,
        snapshot(),
        "source/fixture changed during native replay"
    );
}
