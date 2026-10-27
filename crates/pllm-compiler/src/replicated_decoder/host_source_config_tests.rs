//! Tiny explicit OPERATOR fixtures. Authored only; no production manifest,
//! candidate checksum adoption, producer, inference or BF16 qualification.
use super::*;
use crate::replicated_decoder::program::source_entry::ingress::SourceHeaderEntry;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{cell::Cell, io::Cursor};

const CONFIG: &[u8] = br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":3,"max_position_embeddings":16,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":false,"rope_theta":10000.0}"#;

fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }
fn fixture() -> Value {
    json!({"version":1,"approved_lock_digest":hex(&[2;32]),
        "config_sha256":hex(&Sha256::digest(CONFIG)),"semantic_digest":hex(&[5;32]),"recipe":"D31",
        "files":[{"name":"config.json","size":CONFIG.len(),"sha256":hex(&Sha256::digest(CONFIG))},
                 {"name":"model.safetensors.index.json","size":16,"sha256":hex(&[3;32])},
                 {"name":"synthetic.safetensors","size":1024,"sha256":hex(&[4;32])}],
        "tensors":[{"name":"x.weight","file":2,"dtype":"BF16","shape":[2,2],"begin":0,"end":8,"scale":null}]})
}
fn parse(bytes: &[u8]) -> Result<Startup, Error> {
    let mut state = Startup::new(host_limit()?)?;
    let source = read_source(&state.root, &mut Cursor::new(bytes), bytes.len())?;
    state.parse(source)?;
    Ok(state)
}
fn operator(value: &Value) -> Startup { parse(&serde_json::to_vec(value).unwrap()).unwrap() }
fn experiment() -> Vec<u8> {
    serde_json::to_vec(&json!({"schema":"pllm.experiment.v1","name":"operator-fixture",
        "pipeline":{"model":{"source":"synthetic-original"},"components":{"linear":{
            "component":crate::replicated_decoder::sdk::PROTOCOL,"params":{
                "weights":"public","range_policy":"checked","max_abs_activation":1,"storage":"compact"}}}},
        "deployment":{"kind":"local","root":"."},"budget":{"requests":1,"max_input_tokens":4,"max_new_tokens":2}})).unwrap()
}

// Derive fixture semantics through the existing numerical/canonical owners, not
// a new lock codec or fixture-issued sealed producer. Files remain explicit
// operator assertions: this fixture does not claim any candidate bytes verified.
fn header_fixture() -> Value {
    use pllm_models::{bounded_canonical as canonical, DecoderExpansionBudget, DecoderWorkload};
    use pllm_types::bounded_canonical::{StreamFrame, StreamLimits};
    let plan = pllm_models::lower_model_json_with_budget(CONFIG,
        DecoderWorkload { batch:1, max_input_tokens:4, max_new_tokens:2 },
        DecoderExpansionBudget { max_layers:8, max_feature_width:128, max_attention_heads:8,
            max_vocabulary:128, max_tensor_elements:32768 }).unwrap();
    let semantic = canonical::bounded_plan_digest(&plan, canonical::Limits {
        canonical: StreamLimits { max_nodes:1<<20, max_depth:16, max_bytes:1<<24,
            max_string_bytes:NAME, max_key_bytes:NAME, max_total_string_bytes:1<<20,
            max_work:1<<28, max_stack_bytes:size_of::<[StreamFrame<'_>;16]>() },
        max_operations:DESCRIPTORS*2, max_states:DESCRIPTORS*4, max_inputs:DESCRIPTORS*16,
        max_rank:5, max_transformations:0, max_domain_bytes:pllm_models::DECODER_PLAN_SCHEMA_VERSION.len(),
    }, &mut [StreamFrame::empty();16]).unwrap();
    let (shapes, _) = crate::replicated_decoder::program::source_shapes_from_plan(
        &plan, crate::replicated_decoder::WeightMode::Public).unwrap();
    let mut end = 0u64;
    let tensors: Vec<_> = shapes.into_iter().map(|(name, shape)| {
        let begin = end;
        end += shape.iter().product::<usize>() as u64 * 2;
        json!({"name":name,"file":2,"dtype":"BF16","shape":shape,"begin":begin,"end":end,"scale":null})
    }).collect();
    let mut value = fixture();
    value["semantic_digest"] = json!(hex(&semantic));
    value["tensors"] = json!(tensors);
    value["files"][2]["size"] = json!(end + 1024);
    value
}
fn request(state: &Startup, config: &[u8]) -> Result<(), String> {
    let guard = state.config.as_ref().unwrap().lock().unwrap();
    let retained = state.root.budget().used_bytes();
    guard.with_expected([2;32], |expected| {
        let pending = SourceHeaderEntry::begin_raw_host(&experiment(), config, 4, 2,
            [1;32], [2;32], [3;32], expected).map_err(|error| error.to_string())?;
        // Actual request header/premise copies coexist with the original HOST
        // inventory and scratch guards. Neither migrates to the other's root.
        assert_eq!(state.root.budget().used_bytes(), retained);
        drop(pending);
        Ok(())
    })
}

#[test]
fn absent_startup_is_once_only_unavailable_without_capture_reload() {
    let cell = OnceLock::new();
    let captures = Cell::new(0);
    initialize(&cell, || { captures.set(captures.get()+1); None }).unwrap();
    initialize(&cell, || panic!("startup environment read twice")).unwrap();
    assert_eq!(captures.get(), 1);
    let state = cell.get().unwrap().as_ref().unwrap();
    assert!(state.config.is_none() && state.failure.is_none());
    assert!(state.root.budget().used_bytes() >= control_bytes().unwrap());
}

#[test]
fn strict_json_duplicate_unknown_version_recipe_and_digest_refusals() {
    for bytes in [b"{".as_slice(), b"null", b"[]", b"{}", b"{\"version\":1,\"version\":1}"] {
        assert!(parse(bytes).is_err());
    }
    let original = serde_json::to_string(&fixture()).unwrap();
    let duplicate = original.replacen("{", "{\"version\":1,", 1);
    assert!(parse(duplicate.as_bytes()).is_err());
    let escaped = original.replacen("{", "{\"vers\\u0069on\":1,", 1);
    assert!(parse(escaped.as_bytes()).is_err());
    for (key, bad) in [("version",json!(2)),("recipe",json!("D99")),
        ("approved_lock_digest",json!("00".repeat(32))),
        ("semantic_digest",json!("AA".repeat(32))), ("config_sha256",json!("g".repeat(64))),
        ("revision",json!("labels-never-authority"))] {
        let mut value = fixture(); value[key] = bad;
        assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err(), "{key}");
    }
    for location in ["file", "tensor"] {
        let mut value = fixture();
        if location == "file" { value["files"][0]["extra"] = json!(1); }
        else { value["tensors"][0]["extra"] = json!(1); }
        assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}

#[test]
fn strict_inventory_names_sizes_dtypes_shapes_offsets_and_scales() {
    for (key, bad) in [("name",json!("../x")),("file",json!(1)),("dtype",json!("F64")),
        ("shape",json!([])),("shape",json!([1,1,1])),("shape",json!([0])),
        ("shape",json!([u64::MAX,2])),("begin",json!(u64::MAX)),("end",json!(7)),
        ("end",json!(2048)),("scale",json!("missing_scale"))] {
        let mut value = fixture(); value["tensors"][0][key] = bad;
        assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err(), "{key}");
    }
    for size in [0, 4_412_140_424] {
        let mut value = fixture(); value["files"][2]["size"] = json!(size);
        assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
    }
    assert_eq!(public_file::MAX_BYTES, 4_294_967_296);
    let mut value = fixture(); value["files"][0]["sha256"] = json!(hex(&[9;32]));
    assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
    let mut value = fixture(); value["files"][2]["name"] = json!("config.json");
    assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
    let mut value = fixture();
    let duplicate = value["tensors"][0].clone();
    value["tensors"].as_array_mut().unwrap().push(duplicate);
    assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
    let mut value = fixture(); value["recipe"] = json!("D32");
    value["tensors"][0]["dtype"] = json!("F8_E4M3");
    value["tensors"][0]["end"] = json!(4);
    value["tensors"][0]["scale"] = json!("missing_scale");
    assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
}

#[test]
fn block128_scale_geometry_and_bf16_mapping_use_existing_raw_contract() {
    let mut value = fixture();
    value["recipe"] = json!("D32");
    value["files"][2]["size"] = json!(18000);
    value["tensors"][0] = json!({"name":"x.weight","file":2,"dtype":"F8_E4M3",
        "shape":[128,128],"begin":0,"end":16384,"scale":"x.scale"});
    value["tensors"].as_array_mut().unwrap().push(json!({"name":"x.scale","file":2,
        "dtype":"BF16","shape":[1,1],"begin":16384,"end":16386,"scale":null}));
    // Inventory/relationship only: the operator still owes actual D32 config and
    // semantics. This does not qualify a request or any candidate/producer.
    operator(&value);
    let mut wrong = value.clone(); wrong["tensors"][1]["shape"] = json!([2,1]);
    wrong["tensors"][1]["end"] = json!(16388);
    assert!(parse(&serde_json::to_vec(&wrong).unwrap()).is_err());
    let mut wrong = value.clone(); wrong["tensors"][1]["dtype"] = json!("F32");
    wrong["tensors"][1]["end"] = json!(16388);
    assert!(parse(&serde_json::to_vec(&wrong).unwrap()).is_err());
    let mut wrong = value; wrong["tensors"][0]["scale"] = json!("missing.scale");
    assert!(parse(&serde_json::to_vec(&wrong).unwrap()).is_err());
}

#[test]
fn strict_depth_count_string_and_aggregate_shape_bounds() {
    let mut nested = b"0".to_vec();
    for _ in 0..17 { nested = [b"[".as_slice(), &nested, b"]"].concat(); }
    assert!(parse(&nested).is_err());
    let mut value = fixture(); value["tensors"][0]["name"] = json!("a".repeat(NAME+1));
    assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
    let nodes = format!("[{}]", "0,".repeat(DESCRIPTORS) + "0");
    assert!(parse(nodes.as_bytes()).is_err());
    assert!(parse(&vec![b' '; JSON_BYTES+1]).is_err());
    let mut value = fixture(); value["tensors"][0]["shape"] = json!([1<<30,1<<30]);
    assert!(parse(&serde_json::to_vec(&value).unwrap()).is_err());
}

#[test]
fn exact_identity_selects_same_retained_inventory_foreign_source_never_factory() {
    let state = operator(&fixture());
    let config = state.config.as_ref().unwrap().lock().unwrap();
    let before = state.root.budget().used_bytes();
    let (files, tensors) = config.views().unwrap();
    assert_eq!(files[0].0.as_ptr(), config.files[0].name.as_ptr());
    assert_eq!(tensors[0].0.as_ptr(), config.tensors[0].name.as_ptr());
    assert_eq!(tensors[0].3.as_ptr(), config.tensors[0].shape.as_ptr());
    assert!(config._scratch.payload_bytes() >= scratch_bytes(3,1).unwrap());
    drop((files,tensors));
    config.with_expected([2;32], |expected| { assert!(expected.is_some()); Ok(()) }).unwrap();
    for source in [[0;32],[3;32],Sha256::digest(b"self-consistent-candidate").into()] {
        config.with_expected(source, |expected| {
            if expected.is_some() { panic!("factory/candidate read reached without approved exact identity"); }
            assert!(SourceHeaderEntry::begin_raw_host(b"not parsed", b"not parsed", 4, 2,
                [1;32],source,[3;32],expected).is_err());
            Ok(())
        }).unwrap();
    }
    assert_eq!(state.root.budget().used_bytes(), before);
}

#[test]
fn exact_header_and_actual_configuration_semantics_recipe_inventory_checks() {
    let value = header_fixture();
    let state = operator(&value);
    request(&state, CONFIG).unwrap(); // only pending header, no candidate FS/producer
    let changed = [CONFIG, b" "].concat(); // same JSON semantics, different original bytes
    assert!(request(&state, &changed).is_err());
    let mut bad = value.clone(); bad["semantic_digest"] = json!(hex(&[9;32]));
    assert!(request(&operator(&bad), CONFIG).is_err());
    let mut bad = value.clone(); bad["recipe"] = json!("D32");
    assert!(request(&operator(&bad), CONFIG).is_err());
    let mut bad = value;
    bad["tensors"][0]["name"] = json!("unrelated.weight");
    assert!(request(&operator(&bad), CONFIG).is_err());
}

#[test]
#[cfg(unix)]
fn approved_entry_does_not_turn_candidate_content_into_approved_content() {
    use crate::replicated_decoder::program::source_entry::ingress::RawHost;
    use std::os::unix::fs::MetadataExt;
    // Test-only all-path explicit cleanup. No production finalizer is introduced.
    struct Retire(RawHost);
    impl Drop for Retire {
        fn drop(&mut self) {
            if self.0.retire().is_ok() { self.0.release_final_charges().unwrap(); }
        }
    }
    let state = operator(&header_fixture());
    let guard = state.config.as_ref().unwrap().lock().unwrap();
    let host = Retire(guard.with_expected([2;32], |expected| {
        let header = SourceHeaderEntry::begin_raw_host(&experiment(), CONFIG, 4, 2,
            [1;32], [2;32], [3;32], expected).map_err(|error| error.to_string())?;
        RawHost::prepare(header).map_err(|error| error.to_string())
    }).unwrap());
    let device = host.0.disk_directory().unwrap().metadata().unwrap().dev();
    host.0.start(device).unwrap();
    let mut candidate = CONFIG.to_vec(); candidate[0] ^= 1;
    assert!(host.0.append(0,0,&candidate).is_err()); // whole-file pin still mandatory
    assert_eq!(guard.config_sha256, <[u8;32]>::from(Sha256::digest(CONFIG)));
}

#[test]
fn exact_and_one_short_read_baseline_admit_before_buffer_or_reader() {
    struct Trap;
    impl Read for Trap { fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> { panic!("reader before admission"); } }
    let n = 8usize;
    let required = OwnedBudget::resources().unwrap().initial_reserved_bytes + n + 1 + size_of::<OwnedCharged<Vec<u8>>>();
    let root = OwnedBudget::new(required-1).unwrap();
    assert!(read_source(&root, &mut Trap, n).is_err());
    assert_eq!(root.budget().used_bytes(), OwnedBudget::resources().unwrap().initial_reserved_bytes);
    let root = OwnedBudget::new(required).unwrap();
    let original = read_source(&root, &mut Cursor::new(b"12345678"), n).unwrap();
    assert_eq!(root.budget().used_bytes(), required);
    assert_eq!(original.payload(), b"12345678");
    drop(original);
    assert_eq!(root.budget().used_bytes(), OwnedBudget::resources().unwrap().initial_reserved_bytes);
    let root = OwnedBudget::new(host_limit().unwrap()).unwrap();
    assert!(read_source(&root, &mut Trap, JSON_BYTES+1).is_err());
}

#[test]
fn exact_and_one_short_controls_and_owned_storage_quotes() {
    let controls = OwnedBudget::resources().unwrap().initial_reserved_bytes
        + control_bytes().unwrap() + size_of::<OwnedExternalCharge>();
    assert!(Startup::new(controls-1).is_err());
    assert!(Startup::new(controls).is_ok());
    let storage = storage_bytes(3,1,256).unwrap();
    let scratch = scratch_bytes(3,1).unwrap();
    let total = OwnedBudget::resources().unwrap().initial_reserved_bytes + storage + scratch
        + 2*size_of::<OwnedExternalCharge>();
    let root = OwnedBudget::new(total-1).unwrap();
    let first = root.charge_external(storage).unwrap();
    assert!(root.charge_external(scratch).is_err()); // before ANY view/inventory factory
    drop(first);
    let root = OwnedBudget::new(total).unwrap();
    let first = root.charge_external(storage).unwrap();
    let second = root.charge_external(scratch).unwrap();
    assert_eq!(root.budget().used_bytes(),total);
    drop((first,second));
    assert!(host_limit().unwrap() <= MAX_ALLOCATION_BYTES);
}

#[test]
fn transient_read_parser_and_partial_inventory_retire_but_actual_owner_survives() {
    let state = operator(&fixture());
    let config = state.config.as_ref().unwrap().lock().unwrap();
    let steady = OwnedBudget::resources().unwrap().initial_reserved_bytes
        + state._controls.charged_bytes() + config._storage.charged_bytes() + config._scratch.charged_bytes();
    assert_eq!(state.root.budget().used_bytes(), steady);
    assert!(config.files.iter().all(|file| file.name.capacity() == file.name.len()));
    assert!(config.tensors.iter().all(|tensor| tensor.name.capacity() == tensor.name.len()));
    drop(config);

    let mut state = Startup::new(host_limit().unwrap()).unwrap();
    let before = state.root.budget().used_bytes();
    let bytes = serde_json::to_vec(&fixture()).unwrap();
    let source = read_source(&state.root, &mut Cursor::new(&bytes), bytes.len()).unwrap();
    state.work_left = PARSER_WORK - 1;
    assert!(state.parse(source).is_err()); // before scanner/parser construction
    assert_eq!(state.root.budget().used_bytes(),before);

    let mut bad = fixture(); bad["tensors"][0]["end"] = json!(7);
    let bytes = serde_json::to_vec(&bad).unwrap();
    state.work_left = work_limit().unwrap();
    let source = read_source(&state.root, &mut Cursor::new(&bytes), bytes.len()).unwrap();
    assert!(state.parse(source).is_err());
    assert_eq!(state.root.budget().used_bytes(),before);
}

#[test]
fn oversize_growth_and_truncation_during_read_are_closed() {
    let root = OwnedBudget::new(host_limit().unwrap()).unwrap();
    assert!(read_source(&root, &mut Cursor::new(b"123456789"), 8).is_err());
    assert!(read_source(&root, &mut Cursor::new(b"1234567"), 8).is_err());
    let input = vec![b' '; JSON_BYTES+1];
    assert!(read_source(&root, &mut Cursor::new(input), JSON_BYTES).is_err());
}

#[test]
fn relative_missing_dangling_nonregular_and_oversized_paths_fail_startup_closed() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing.json");
    let oversized = directory.path().join("oversized.json");
    File::create(&oversized).unwrap().set_len((JSON_BYTES+1) as u64).unwrap();
    let malformed = directory.path().join("malformed.json");
    std::fs::write(&malformed,b"{").unwrap();
    let empty = directory.path().join("empty.json");
    std::fs::write(&empty,b"").unwrap();
    for path in [OsString::from("relative.json"), missing.into_os_string(),
        directory.path().as_os_str().to_owned(), oversized.into_os_string(), malformed.into_os_string(),
        empty.into_os_string(), OsString::from("/a\0b"), OsString::from("/".to_owned()+&"a".repeat(PATH_BYTES))] {
        let cell = OnceLock::new();
        assert!(initialize(&cell, || Some(path)).is_err());
        initialize(&cell, || panic!("failed startup recaptured environment")).unwrap_err();
        let state = cell.get().unwrap().as_ref().unwrap();
        assert!(state.config.is_none() && state.failure.is_some());
        let retained_path = state.path.as_ref().map_or(0, OwnedCharged::charged_bytes);
        assert_eq!(state.root.budget().used_bytes(),
            OwnedBudget::resources().unwrap().initial_reserved_bytes + state._controls.charged_bytes() + retained_path);
    }
    #[cfg(unix)] {
        let link = directory.path().join("dangling.json");
        std::os::unix::fs::symlink(directory.path().join("absent"), &link).unwrap();
        assert!(initialize(&OnceLock::new(), || Some(link.into_os_string())).is_err());
    }
}

#[test]
fn initialized_inventory_ignores_later_environment_capture_file_and_candidate_changes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("operator.json");
    std::fs::write(&path, serde_json::to_vec(&fixture()).unwrap()).unwrap();
    let cell = OnceLock::new();
    initialize(&cell, || Some(path.clone().into_os_string())).unwrap();
    std::fs::write(&path, b"malformed replacement").unwrap();
    // Pure capture substitution models env mutation WITHOUT mutating process env.
    initialize(&cell, || panic!("environment re-read/reload attempted")).unwrap();
    let state = cell.get().unwrap().as_ref().unwrap();
    let config = state.config.as_ref().unwrap().lock().unwrap();
    config.with_expected([2;32], |expected| { assert!(expected.is_some()); Ok(()) }).unwrap();
    let candidate = serde_json::to_vec(&json!({"approved_lock_digest":hex(&[7;32]),"config_sha256":hex(&[9;32])})).unwrap();
    config.with_expected(Sha256::digest(&candidate).into(), |expected| {
        assert!(expected.is_none()); // self-consistent candidate is not an operator entry
        Ok(())
    }).unwrap();
    assert_eq!(config.approved_lock_digest,[2;32]);
    assert_eq!(config.config_sha256, <[u8;32]>::from(Sha256::digest(CONFIG)));
}

#[test]
fn count_quoted_validation_work_exact_and_one_short_on_original_startup_meter() {
    let manifest = header_fixture();
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let validation = validation_work(manifest["files"].as_array().unwrap().len(),
        manifest["tensors"].as_array().unwrap().len()).unwrap();
    let prefix = PARSER_WORK + 2 * DESCRIPTORS as u128;
    let required = prefix + validation;
    for short in [false, true] {
        let mut state = Startup::new(host_limit().unwrap()).unwrap();
        let before = state.root.budget().used_bytes();
        let original = state.work_left;
        // Narrow the ORIGINAL startup allowance; no replacement ledger, MAX
        // grant, invalid schema or public/operator-injected work limit.
        assert!(required <= original);
        state.pay(original - required + u128::from(short)).unwrap();
        let admitted = state.work_left;
        assert_eq!(admitted, required - u128::from(short));
        let source = read_source(&state.root, &mut Cursor::new(&bytes), bytes.len()).unwrap();
        let output = state.parse(source);
        if short {
            assert_eq!(output, Err("host logical work exhausted"));
            assert_eq!(admitted - state.work_left, prefix);
            assert_eq!(state.work_left, validation - 1); // failed debit spends nothing
            assert_eq!(state.validation_builds, 0); // no inventory/duplicate/scale/coverage walk
            assert!(state.config.is_none()); // no output, header, factory or candidate path
            assert_eq!(state.root.budget().used_bytes(), before); // read/parser owners retired
        } else {
            output.unwrap();
            assert_eq!(admitted - state.work_left, required);
            assert_eq!(state.work_left, 0);
            assert_eq!(state.validation_builds, 1);
            assert!(state.config.is_some());
            request(&state, CONFIG).unwrap(); // same generated manifest passes actual premise
            drop(state.config.take());
            assert_eq!(state.root.budget().used_bytes(), before);
        }
    }
}

#[test]
fn concurrent_borrowed_callbacks_serialize_same_entry_and_keep_permanent_charges() {
    use std::sync::{atomic::{AtomicUsize, Ordering}, mpsc, TryLockError};
    let mut state = operator(&header_fixture());
    let mutex = state.config.as_ref().unwrap();
    let (identity, file_name, tensor_name, shape, storage, scratch) = {
        let guard = mutex.lock().unwrap();
        (&*guard as *const HostSourceConfig as usize, guard.files[0].name.as_ptr() as usize,
            guard.tensors[0].name.as_ptr() as usize, guard.tensors[0].shape.as_ptr() as usize,
            guard._storage.charged_bytes(), guard._scratch.charged_bytes())
    };
    let steady = state.root.budget().used_bytes();
    assert_eq!(steady, OwnedBudget::resources().unwrap().initial_reserved_bytes
        + state._controls.charged_bytes() + storage + scratch);
    let check_owner = |guard: &HostSourceConfig| {
        assert_eq!(guard as *const HostSourceConfig as usize, identity);
        assert_eq!(guard.approved_lock_digest, [2;32]);
        assert_eq!(guard.files[0].name.as_ptr() as usize, file_name);
        assert_eq!(guard.tensors[0].name.as_ptr() as usize, tensor_name);
        assert_eq!(guard.tensors[0].shape.as_ptr() as usize, shape);
        assert_eq!(guard._storage.charged_bytes(), storage);
        assert_eq!(guard._scratch.charged_bytes(), scratch);
        assert_eq!(state.root.budget().used_bytes(), steady);
    };
    let phase = AtomicUsize::new(0);
    let (entered_tx, entered_rx) = mpsc::sync_channel(0);
    let (blocked_tx, blocked_rx) = mpsc::sync_channel(0);
    // Exactly two scoped workers. All endpoints move to their worker, so failure
    // drops the peer's channel rather than stranding a recv. Join BOTH before
    // asserting either result; no sleeps, timeout/scheduling proof or global cell.
    let (first, second) = std::thread::scope(|scope| {
        let check_owner = &check_owner;
        let phase = &phase;
        let first = scope.spawn(move || {
            let guard = mutex.lock().unwrap(); // same local Startup mutex as production lookup
            let result: Result<(), String> = guard.with_expected([2;32], |expected| {
                check_owner(&guard);
                assert!(expected.is_some()); // keep borrowed expectation alive across handshake
                assert_eq!(phase.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst), Ok(0));
                entered_tx.send(()).unwrap();
                blocked_rx.recv().unwrap(); // peer has actually observed WouldBlock
                let pending = SourceHeaderEntry::begin_raw_host(&experiment(), CONFIG, 4, 2,
                    [1;32], [2;32], [3;32], expected).unwrap();
                check_owner(&guard); // views + actual header copies still overlap owner/charges
                assert_eq!(phase.compare_exchange(1, 2, Ordering::SeqCst, Ordering::SeqCst), Ok(1));
                drop(pending);
                Err("intentional callback refusal".to_owned())
            });
            check_owner(&guard); // callback error did not retire inventory or scratch
            drop(guard); // only now may second lock/callback enter
            result
        });
        let second = scope.spawn(move || {
            entered_rx.recv().unwrap();
            assert!(matches!(mutex.try_lock(), Err(TryLockError::WouldBlock)));
            assert_eq!(phase.load(Ordering::SeqCst), 1);
            blocked_tx.send(()).unwrap();
            let guard = mutex.lock().unwrap();
            guard.with_expected([2;32], |expected| {
                assert_eq!(phase.compare_exchange(2, 3, Ordering::SeqCst, Ordering::SeqCst), Ok(2));
                check_owner(&guard);
                let pending = SourceHeaderEntry::begin_raw_host(&experiment(), CONFIG, 4, 2,
                    [1;32], [2;32], [3;32], expected).unwrap();
                check_owner(&guard);
                drop(pending);
                Ok(())
            })
        });
        (first.join(), second.join())
    });
    assert_eq!(first.unwrap(), Err("intentional callback refusal".to_owned()));
    assert_eq!(second.unwrap(), Ok(()));
    assert_eq!(phase.load(Ordering::SeqCst), 3);
    check_owner(&mutex.lock().unwrap());
    drop(state.config.take()); // both callbacks/views retired; guards now retire normally
    assert_eq!(state.root.budget().used_bytes(), steady - storage - scratch);
}
