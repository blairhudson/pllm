use super::*;
use crate::replicated_decoder::{
    program, ExecutionPolicy, Manifest, Program, PublicSourceImport, SourceNorms, WeightMode,
};
use pllm_core::puma;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const CONFIG: &[u8] = br#"{"model_type":"llama","hidden_size":2,"intermediate_size":2,
    "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
    "vocab_size":3,"max_position_embeddings":64,"rope_theta":10000,
    "rms_norm_eps":0.000001,"tie_word_embeddings":false,"hidden_act":"silu",
    "attention_bias":false,"head_dim":2,"attention_dropout":0.0,"use_cache":true}"#;
const KEY: &str = "model.embed_tokens.weight";

// Test expectations use explicit raw Q18 words, not encode_q18 or scanner
// Summary authority. Every manifest claim must then survive genuine import.
fn fixture(selected: [i64; 6]) -> (Manifest, BTreeMap<String, Vec<i64>>) {
    let execution = ExecutionPolicy::default();
    let shapes = Program::source_shapes_with_policy(CONFIG, 1, 1, WeightMode::Public, execution)
        .unwrap();
    let plan = program::lower_with_weights(CONFIG, 1, 1, WeightMode::Public).unwrap();
    let inline: std::collections::BTreeSet<_> = plan.prefill.operations.iter()
        .filter(|op| op.operator == pllm_models::ModelOperator::RmsNorm)
        .map(|op| op.attributes["weight"].as_str().unwrap())
        .collect();
    let mut tensors = BTreeMap::new();
    let mut words = BTreeMap::new();
    for (key, shape) in shapes {
        let values = if key == KEY { selected.to_vec() } else { vec![0; shape.iter().product()] };
        assert_eq!(values.len(), shape.iter().product::<usize>());
        let mut row_l1 = None;
        let mut row_squared_norm = None;
        let mut row_l1_squared_sum = None;
        if shape.len() == 2 {
            let sums: Vec<u128> = values.chunks_exact(shape[1])
                .map(|row| row.iter().map(|&n| i128::from(n).unsigned_abs()).sum())
                .collect();
            row_l1 = Some(u64::try_from(*sums.iter().max().unwrap()).unwrap());
            row_squared_norm = Some(values.chunks_exact(shape[1])
                .map(|row| row.iter().map(|&n| i128::from(n).unsigned_abs().pow(2)).sum::<u128>())
                .max().unwrap());
            row_l1_squared_sum = Some(sums.iter().map(|n| n * n).sum());
        }
        let encoded: Vec<u8> = values.iter().flat_map(|n| n.to_le_bytes()).collect();
        tensors.insert(key.clone(), SourceTensor {
            shape,
            lower: *values.iter().min().unwrap(),
            upper: *values.iter().max().unwrap(),
            public_q18: inline.contains(key.as_str()).then(|| values.clone()),
            digest: Sha256::digest(encoded).into(),
            row_l1,
            row_squared_norm,
            row_l1_squared_sum,
        });
        words.insert(key, values);
    }
    (Manifest {
        schema: "pllm.replicated-decoder.manifest.v1".into(),
        weights: WeightMode::Public,
        execution,
        config: CONFIG.to_vec(),
        rows: 1,
        outputs: 1,
        source: [1; 32], composition: [2; 32], session: [3; 32],
        tensors,
        norms: SourceNorms::new(),
        program: [0; 32], // No expanded numeric admission or execution.
    }, words)
}

fn feed(importer: &mut PublicSourceImport, words: &BTreeMap<String, Vec<i64>>)
    -> Result<(), String>
{
    for (key, values) in words {
        // Three-word frames split width-two rows. All fixture words are exactly
        // representable after scaling, including tiny raw 8 and raw 2^36.
        for (frame, chunk) in values.chunks(3).enumerate() {
            let bytes: Vec<u8> = chunk.iter()
                .flat_map(|&n| ((n as f32) / 262144.0).to_le_bytes()).collect();
            importer.append(key, frame * 3, &bytes)?;
        }
    }
    Ok(())
}

fn import(manifest: &Manifest, words: &BTreeMap<String, Vec<i64>>) -> PublicSource {
    let mut importer = PublicSourceImport::new(manifest).unwrap();
    feed(&mut importer, words).unwrap();
    importer.finish().unwrap()
}

#[test]
fn mixed_nonsquare_rows_survive_real_split_frames_and_preserve_owner_layouts() {
    let (manifest, words) = fixture([8, -3, -2, 1, 4, -9]);
    let serialized = serde_json::to_vec(&manifest).unwrap();
    let binding = program::source_binding_with_norms(&manifest.tensors, &manifest.norms);
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let source = import(&manifest, &words);
    let lookup = KEY.to_owned();
    let cap = StoredRowDotCap::new(&source, &registry, &lookup, 100).unwrap().unwrap();
    assert_eq!(cap.stored_shape(), &[3, 2]);
    assert_eq!(cap.stored_row_l1(), 13);
    assert_eq!(cap.key(), KEY);
    assert_eq!(cap.requested(), 100);
    assert_eq!(cap.candidate_cap(), 100);
    let (actual_key, actual_record) = registry.source_records().get_key_value(KEY).unwrap();
    assert!(std::ptr::eq(cap.source, &source));
    assert!(std::ptr::eq(cap.registry, &registry));
    assert!(std::ptr::eq(cap.key, actual_key));
    assert!(std::ptr::eq(cap.record, actual_record));
    assert!(!std::ptr::eq(cap.key, &lookup));
    assert_eq!(registry.owned_backing_bytes(), 0);
    assert_eq!(serde_json::to_vec(&manifest).unwrap(), serialized);
    assert_eq!(registry.binding(), binding);
    source.check_registry(&registry).unwrap();

    let (second_manifest, second_words) = fixture([8, -3, -2, 1, 4, -9]);
    let second_registry = SourceRegistry::from_manifest(&second_manifest).unwrap();
    let second_source = import(&second_manifest, &second_words);
    assert_eq!(second_registry.binding(), binding);
    assert!(!std::ptr::eq(cap.source, &second_source));
    assert!(!std::ptr::eq(cap.registry, &second_registry));
    assert!(!std::ptr::eq(cap.record, &second_manifest.tensors[KEY]));
    // Equal-content imports may construct independent leaves. There is no API
    // to reattach the original leaf to either new owner.
    let second = StoredRowDotCap::new(&second_source, &second_registry, KEY, 100)
        .unwrap().unwrap();
    assert!(std::ptr::eq(second.source, &second_source));
    assert!(std::ptr::eq(second.record, &second_manifest.tensors[KEY]));
}

#[test]
fn stored_row_orientation_cannot_supply_transposed_column_premise() {
    let (manifest, words) = fixture([8, 0, 8, 0, 8, 0]);
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let source = import(&manifest, &words);
    let request = (1_i64 << 59) - 1;
    let cap = StoredRowDotCap::new(&source, &registry, KEY, request).unwrap().unwrap();
    assert_eq!(cap.stored_shape(), &[3, 2]);
    assert_eq!(cap.stored_row_l1(), 8);
    assert_eq!(cap.candidate_cap(), request);
    let domain = i128::from(puma::INPUT_MAX_EXCLUSIVE);
    let row_abs: Vec<i128> = words[KEY].chunks_exact(2)
        .map(|row| row.iter().map(|&n| i128::from(n).abs()).sum()).collect();
    assert_eq!(row_abs, vec![8, 8, 8]);
    let column_abs: i128 = words[KEY].chunks_exact(2).map(|row| i128::from(row[0]).abs()).sum();
    assert_eq!(column_abs, 24);
    assert!(i128::from(cap.candidate_cap()) * 8 < domain);
    assert!(i128::from(cap.candidate_cap()) * column_abs >= domain);
}

#[test]
fn real_q18_boundary_excludes_endpoint_for_every_signed_product_and_prefix() {
    let (manifest, words) = fixture([1_i64 << 36, 0, 0, 0, 0, 0]);
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let source = import(&manifest, &words);
    let request = 1_i64 << 26;
    let cap = StoredRowDotCap::new(&source, &registry, KEY, request).unwrap().unwrap();
    assert_eq!(cap.stored_row_l1(), 1_u64 << 36);
    assert_eq!(cap.candidate_cap(), request - 1);
    let domain = i128::from(puma::INPUT_MAX_EXCLUSIVE);
    assert_eq!(i128::from(request) * (1_i128 << 36), domain);
    for row in words[KEY].chunks_exact(2) {
        for mask in 0..4 {
            let mut prefix = 0_i128;
            for (index, &weight) in row.iter().enumerate() {
                let operand = if mask & (1 << index) == 0 { cap.candidate_cap() } else { -cap.candidate_cap() };
                let product = i128::from(operand) * i128::from(weight);
                prefix += product;
                assert!(product > -domain && product < domain);
                assert!(prefix > -domain && prefix < domain);
            }
        }
    }
}

#[test]
fn authenticated_absent_l1_is_not_authenticated_zero() {
    let (mut manifest, words) = fixture([0; 6]);
    let zero_registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let zero_source = import(&manifest, &words);
    let zero = StoredRowDotCap::new(&zero_source, &zero_registry, KEY, i64::MAX)
        .unwrap().unwrap();
    assert_eq!(zero.stored_row_l1(), 0);
    assert_eq!(zero.candidate_cap(), i64::MAX);
    assert!(matches!(StoredRowDotCap::new(&zero_source, &zero_registry, KEY, -1), Err(Error::Bounds)));
    // Leave an independently authenticated squared norm present: it must not
    // manufacture an L1 premise when optional L1/energy are absent.
    let tensor = manifest.tensors.get_mut(KEY).unwrap();
    tensor.row_l1 = None;
    tensor.row_l1_squared_sum = None;
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let source = import(&manifest, &words);
    assert!(StoredRowDotCap::new(&source, &registry, KEY, i64::MAX).unwrap().is_none());
    assert!(matches!(StoredRowDotCap::new(&source, &registry, KEY, -1), Err(Error::Bounds)));
    assert!(matches!(StoredRowDotCap::new(&zero_source, &registry, "missing", -1), Err(Error::Bounds)));
    assert!(matches!(StoredRowDotCap::new(&zero_source, &registry, "missing", 0), Err(Error::Binding)));
    assert!(matches!(StoredRowDotCap::new(&source, &registry, "missing", 0), Err(Error::Binding)));
    let gamma = manifest.tensors.iter().find(|(_, tensor)| tensor.public_q18.is_some()).unwrap().0;
    assert!(matches!(StoredRowDotCap::new(&source, &registry, gamma, 0), Err(Error::Shape)));
}

#[test]
fn legal_looking_statistic_tamper_fails_old_binding_and_real_reimport() {
    let (mut manifest, words) = fixture([8, -3, -2, 1, 4, -9]);
    let source = import(&manifest, &words);
    manifest.tensors.get_mut(KEY).unwrap().row_l1 = Some(14);
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    assert!(matches!(StoredRowDotCap::new(&source, &registry, KEY, 1), Err(Error::Binding)));
    let mut importer = PublicSourceImport::new(&manifest).unwrap();
    assert_eq!(feed(&mut importer, &words).unwrap_err(), "public source row bound differs");
    assert!(importer.finish().is_err());
}

#[test]
fn changed_digest_or_real_coefficient_bytes_cannot_substitute_source_provenance() {
    let (mut manifest, words) = fixture([8, -3, -2, 1, 4, -9]);
    let source = import(&manifest, &words);
    manifest.tensors.get_mut(KEY).unwrap().digest[0] ^= 1;
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    assert!(matches!(
        StoredRowDotCap::new(&source, &registry, KEY, 1),
        Err(Error::Binding)
    ));
    let mut importer = PublicSourceImport::new(&manifest).unwrap();
    assert_eq!(feed(&mut importer, &words).unwrap_err(), "public source digest differs");
    assert!(importer.finish().is_err());

    // Sign changes preserve all three row statistics and the interval. Only the
    // actual ordered encoded-byte digest distinguishes these real imports.
    let (original_manifest, original_words) = fixture([8, -3, -2, 1, 4, -9]);
    let (foreign_manifest, foreign_words) = fixture([8, 3, -2, 1, 4, -9]);
    let original_registry = SourceRegistry::from_manifest(&original_manifest).unwrap();
    let foreign_source = import(&foreign_manifest, &foreign_words);
    assert_eq!(original_manifest.tensors[KEY].row_l1, foreign_manifest.tensors[KEY].row_l1);
    assert_eq!(original_manifest.tensors[KEY].row_squared_norm, foreign_manifest.tensors[KEY].row_squared_norm);
    assert_eq!(original_manifest.tensors[KEY].row_l1_squared_sum, foreign_manifest.tensors[KEY].row_l1_squared_sum);
    assert_ne!(original_manifest.tensors[KEY].digest, foreign_manifest.tensors[KEY].digest);
    assert!(matches!(
        StoredRowDotCap::new(&foreign_source, &original_registry, KEY, 1),
        Err(Error::Binding)
    ));
    let mut contradictory = original_words;
    contradictory.get_mut(KEY).unwrap()[1] = 3;
    let mut importer = PublicSourceImport::new(&original_manifest).unwrap();
    assert_eq!(feed(&mut importer, &contradictory).unwrap_err(), "public source digest differs");
    assert!(importer.finish().is_err());
}

#[test]
fn incomplete_import_bad_geometry_and_missing_authenticated_files_never_issue_leaf() {
    let (manifest, words) = fixture([8, -3, -2, 1, 4, -9]);
    assert!(PublicSourceImport::new(&manifest).unwrap().finish().is_err());
    let mut partial = PublicSourceImport::new(&manifest).unwrap();
    let (first_key, first_words) = words.first_key_value().unwrap();
    let raw = ((first_words[0] as f32) / 262144.0).to_le_bytes();
    partial.append(first_key, 0, &raw).unwrap();
    assert!(partial.finish().is_err());
    for shape in [vec![6], vec![0, 2], vec![3, 0], vec![usize::MAX, 2]] {
        let (mut malformed, _) = fixture([0; 6]);
        malformed.tensors.get_mut(KEY).unwrap().shape = shape;
        assert!(SourceRegistry::from_manifest(&malformed).is_err());
        assert!(PublicSourceImport::new(&malformed).is_err());
    }
    let registry = SourceRegistry::from_manifest(&manifest).unwrap();
    let mut source = import(&manifest, &words);
    let _original_files = source.take_files_for_tests();
    assert!(matches!(StoredRowDotCap::new(&source, &registry, KEY, 1), Err(Error::Binding)));
}
