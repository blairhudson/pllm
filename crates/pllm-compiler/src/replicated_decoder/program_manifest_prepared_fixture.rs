//! Opt-in test fixture for one ORIGINAL public-projection rescale capability.
//!
//! Default expanded lowering uses a closure, not PreparedTensorMap. While this
//! fixture is armed, the selected projection constructs its equivalent closed
//! capability ONCE at that original factory site and publishes those exact Arcs.
//! No already compiled kernel is wrapped, regenerated, or digest-substituted.
//! The returned Program comes from the EXACT Manifest::compile call below,
//! including ordinary manifest identity validation/binding and final retirement.
//! This is not complete compact execution or a production admission/dispatch API.
use super::*;
use crate::replicated_decoder::prepared_tensor_map::{OneLoopScope, PreparedTensorMap};
use std::cell::{Cell, RefCell};

// Original layer-zero Qwen2 Linear weight attribute, preserved by Publisher.
pub(in crate::replicated_decoder) const Q_PROJECTION_SOURCE_KEY: &str =
    "model.layers.0.self_attn.q_proj.weight";

struct Captured {
    prepared: PreparedTensorMap,
    input: usize,
    output: usize,
}

struct Capture {
    source_key: &'static str,
    captured: Option<Captured>,
}

thread_local! {
    static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) };
    static PUBLICATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CaptureGuard;
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        // Also drop a captured capability on compiler/manifest failure or panic.
        CAPTURE.with(|state| *state.borrow_mut() = None);
    }
}

/// Only called by program.rs's cfg(test) public_linear capture point. Unarmed,
/// already satisfied, private-weight, or other-source calls do NO construction.
pub(super) fn capture_public_rescale(
    builder: &mut Builder,
    weight: usize,
    input: usize,
    lanes: usize,
    bounds: Interval,
) -> Result<Option<usize>, Error> {
    let selected = CAPTURE.with(|state| {
        let state = state.borrow();
        let Some(capture) = state.as_ref() else {
            return false;
        };
        capture.captured.is_none()
            && matches!(builder.nodes.get(weight).map(|n| &n.code),
                Some(Code::PublicInput { key, .. }) if key.as_str() == capture.source_key)
    });
    if !selected {
        return Ok(None);
    }
    // Same rescale equation/width/domain as the original closure. The existing
    // prepared factory preadmits full/tail construction with retained Builder.
    let prepared = PreparedTensorMap::rescale(lanes, bounds, 18, builder.allocated, |peak| {
        if peak > MAX_PROGRAM_BYTES {
            Err(Error::Bounds)
        } else {
            Ok(())
        }
    })?;
    // Existing publication rejects equal-digest DIFFERENT allocations in the
    // interner. Never silently substitute an older Arc or retry another factory.
    let output = builder.prepared_tensor_map(input, &prepared)?;
    CAPTURE.with(|state| {
        let mut state = state.borrow_mut();
        let capture = state.as_mut().ok_or(Error::Binding)?;
        if capture.captured.is_some() {
            return Err(Error::Binding);
        }
        capture.captured = Some(Captured {
            prepared,
            input,
            output,
        });
        Ok(())
    })?;
    PUBLICATIONS.with(|count| count.set(count.get() + 1));
    #[cfg(test)]
    tests::after_capture();
    Ok(Some(output))
}

/// Capture only within THIS invocation, never Publisher::seal's earlier compile.
/// Failed final manifest verification returns no Program/capability. The fixture
/// is thread-local, non-reentrant and retains at most one capability, not a trace
/// or a per-node/kernel collection. All production compilation stays unchanged.
pub(in crate::replicated_decoder) fn compile_manifest_rescale18(
    manifest: &Manifest,
    source_key: &'static str,
) -> Result<(Arc<Program>, PreparedTensorMap, usize, usize), String> {
    CAPTURE.with(|state| {
        let mut state = state.borrow_mut();
        if state.is_some() {
            return Err("manifest prepared capture already active".to_string());
        }
        *state = Some(Capture {
            source_key,
            captured: None,
        });
        Ok(())
    })?;
    let _guard = CaptureGuard;
    let program = manifest.compile()?;
    let capture = CAPTURE
        .with(|state| state.borrow_mut().take())
        .ok_or("manifest prepared capture disappeared")?;
    let Captured {
        prepared,
        input,
        output,
    } = capture
        .captured
        .ok_or("manifest has no selected public-projection rescale")?;
    let node = program.nodes.get(output).ok_or("captured output absent")?;
    let Code::TensorLoop {
        input: LoopInput::Slice(owner),
        program: full,
        tail,
    } = &node.code
    else {
        return Err("captured output is not original Slice TensorLoop".into());
    };
    if *owner != input
        || prepared.one_loop_scope() != Some(OneLoopScope::Rescale18)
        || node.lanes != prepared.lanes()
        || !Arc::ptr_eq(full, prepared.program_at(0).map_err(|e| e.to_string())?)
        || output + 1 >= program.nodes.len()
    {
        return Err("captured original mapping differs".into());
    }
    let remainder = node.lanes % full.lanes();
    match (remainder, tail) {
        (0, None) => (),
        (0, Some(_)) | (_, None) => return Err("captured tail shape differs".into()),
        (_, Some(tail)) => {
            if !Arc::ptr_eq(
                tail,
                prepared
                    .program_at(node.lanes - remainder)
                    .map_err(|e| e.to_string())?,
            ) {
                return Err("captured original tail differs".into());
            }
        }
    }
    Ok((program, prepared, input, output))
}

/// Tiny REAL public-source manifest, built/sealed through the ordinary publisher.
/// Its earlier compile is deliberately unarmed. Nonzero Q projection gives the
/// selected rescale a nonzero declared domain rather than reusing a zero-domain
/// kernel already interned by lookup. Actual token input is zero, not embedding.
pub(in crate::replicated_decoder) fn public_manifest() -> Result<Manifest, String> {
    let config = br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,"vocab_size":3,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":true,"max_position_embeddings":16,"rope_theta":10000,"rms_norm_eps":0.000001}"#;
    let execution = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        max_abs_activation: 1,
        ..ExecutionPolicy::default()
    };
    let mut publisher = Publisher::new_with_policy(
        config,
        1,
        1,
        [1; 32],
        [2; 32],
        WeightMode::Public,
        execution,
    )?;
    if !publisher
        .shapes()
        .iter()
        .any(|(key, _)| key == Q_PROJECTION_SOURCE_KEY)
    {
        return Err("fixture Q projection source absent".into());
    }
    let shapes = publisher.shapes().to_vec();
    for (key, shape) in shapes {
        let value = if key.contains("norm") {
            1f32
        } else if key == "model.embed_tokens.weight" {
            0.25f32
        } else if key == Q_PROJECTION_SOURCE_KEY {
            0.09375f32
        } else {
            0f32
        };
        let bytes: Vec<_> = (0..shape.iter().product::<usize>())
            .flat_map(|_| value.to_le_bytes())
            .collect();
        publisher.append(&key, 0, &bytes)?;
    }
    publisher.seal(&[0])?;
    serde_json::from_slice(&publisher.manifest()?).map_err(|e| e.to_string())
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)] // Additional geometry helpers share this test-only fixture.
mod tests {
    use super::*;

    struct CaptureProbe {
        owner: Option<std::sync::Weak<tensor::Program>>,
        unwind: bool,
    }

    thread_local! {
        static PROBE: RefCell<Option<CaptureProbe>> = const { RefCell::new(None) };
    }

    // Observe only the original published owner, never construct or substitute a
    // kernel. Both this hook and its opt-in state live inside the test fixture.
    pub(super) fn after_capture() {
        let unwind = PROBE.with(|probe| {
            let mut probe = probe.borrow_mut();
            let Some(probe) = probe.as_mut() else {
                return false;
            };
            CAPTURE.with(|state| {
                let state = state.borrow();
                let captured = state.as_ref().unwrap().captured.as_ref().unwrap();
                assert_eq!(captured.prepared.programs().count(), 1);
                probe.owner = Some(Arc::downgrade(captured.prepared.program_at(0).unwrap()));
            });
            probe.unwind
        });
        // No RefCell borrow survives the panic: CaptureGuard must clean up while
        // the original compiler Builder and its interner unwind normally.
        if unwind {
            panic!("test unwind after original manifest capture");
        }
    }

    fn large_stack(run: fn()) {
        std::thread::Builder::new()
            .stack_size(16 << 20)
            .spawn(run)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn exact_manifest_call_preserves_original_arc_binding_schedule_and_retirement() {
        large_stack(original_capture);
    }

    fn original_capture() {
        let manifest = public_manifest().unwrap();
        assert!(CAPTURE.with(|state| state.borrow().is_none()));
        let ordinary = manifest.compile().unwrap();
        let before = PUBLICATIONS.with(Cell::get);
        let (program, prepared, input, output) =
            compile_manifest_rescale18(&manifest, Q_PROJECTION_SOURCE_KEY).unwrap();
        assert_eq!(PUBLICATIONS.with(Cell::get), before + 1);
        assert!(CAPTURE.with(|state| state.borrow().is_none()));
        assert_eq!(program.digest(), manifest.program);
        assert_eq!(program.digest(), ordinary.digest());
        assert_eq!(program.source_binding, ordinary.source_binding);
        assert_eq!(program.nodes.len(), ordinary.nodes.len());
        assert_eq!(program.outputs, ordinary.outputs);
        assert_eq!(program.retire, ordinary.retire);
        assert_eq!(program.payload_bytes(), ordinary.payload_bytes());
        assert_eq!(program.work_budget(), ordinary.work_budget());
        assert_eq!(
            program.work_budget(),
            schedule_work(&program.nodes).unwrap()
        );
        let node = &program.nodes[output];
        assert!(matches!(
            program.nodes[input].code,
            Code::PublicProduct { .. }
        ));
        assert_eq!(node.lanes, prepared.lanes());
        assert_eq!(node.bounds, prepared.output_bounds().unwrap());
        assert_eq!(node.work_budget().unwrap(), prepared.work_budget().unwrap());
        let Code::TensorLoop {
            program: full,
            tail,
            ..
        } = &node.code
        else {
            unreachable!()
        };
        let Code::TensorLoop {
            program: original,
            tail: original_tail,
            ..
        } = &ordinary.nodes[output].code
        else {
            unreachable!()
        };
        assert!(Arc::ptr_eq(full, prepared.program_at(0).unwrap()));
        assert_eq!(full.digest(), original.digest());
        assert!(!Arc::ptr_eq(full, original));
        assert_eq!(tail.is_some(), original_tail.is_some());
        assert!(program.retire[output].contains(&input));
        let before = super::super::super::prepared_tensor_map::native_attempts();
        prepared.validate().unwrap();
        assert_eq!(
            super::super::super::prepared_tensor_map::native_attempts(),
            before
        );
    }

    #[test]
    fn capture_is_scoped_and_failed_manifest_verification_never_leaks_a_capability() {
        large_stack(scoped_capture);
    }

    #[test]
    fn failed_manifest_and_unwind_after_capture_destroy_original_kernel_owners() {
        large_stack(failed_capture_owners);
    }

    fn failed_capture_owners() {
        let mut manifest = public_manifest().unwrap();
        for unwind in [false, true] {
            if !unwind {
                manifest.program[0] ^= 1;
            }
            PROBE.with(|probe| {
                *probe.borrow_mut() = Some(CaptureProbe {
                    owner: None,
                    unwind,
                });
            });
            let before = PUBLICATIONS.with(Cell::get);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compile_manifest_rescale18(&manifest, Q_PROJECTION_SOURCE_KEY)
            }));
            if unwind {
                assert!(result.is_err());
            } else {
                assert!(matches!(result, Ok(Err(_))));
                manifest.program[0] ^= 1;
            }
            assert_eq!(PUBLICATIONS.with(Cell::get), before + 1);
            assert!(CAPTURE.with(|state| state.borrow().is_none()));
            let owner = PROBE.with(|probe| probe.borrow_mut().take().unwrap().owner.unwrap());
            assert_eq!(owner.strong_count(), 0);
            assert!(owner.upgrade().is_none());
            // Neither a failed final digest gate nor unwinding poisons the next
            // exact Manifest::compile call on this same thread.
            let (program, prepared, _, _) =
                compile_manifest_rescale18(&manifest, Q_PROJECTION_SOURCE_KEY).unwrap();
            assert_eq!(program.digest(), manifest.program);
            drop((program, prepared));
        }
    }

    fn scoped_capture() {
        let mut manifest = public_manifest().unwrap();
        manifest.program[0] ^= 1;
        let before = PUBLICATIONS.with(Cell::get);
        assert!(compile_manifest_rescale18(&manifest, Q_PROJECTION_SOURCE_KEY).is_err());
        // Capture really occurred before the FINAL Manifest::compile digest gate.
        assert_eq!(PUBLICATIONS.with(Cell::get), before + 1);
        assert!(CAPTURE.with(|state| state.borrow().is_none()));
        manifest.program[0] ^= 1;
        assert!(compile_manifest_rescale18(&manifest, "absent.weight").is_err());
        assert!(CAPTURE.with(|state| state.borrow().is_none()));
        let (program, prepared, _, _) =
            compile_manifest_rescale18(&manifest, Q_PROJECTION_SOURCE_KEY).unwrap();
        assert_eq!(program.digest(), manifest.program);
        drop((program, prepared));
        CAPTURE.with(|state| {
            *state.borrow_mut() = Some(Capture {
                source_key: "outer",
                captured: None,
            })
        });
        let guard = CaptureGuard;
        let before = super::super::super::prepared_tensor_map::native_attempts();
        assert!(compile_manifest_rescale18(&manifest, Q_PROJECTION_SOURCE_KEY).is_err());
        assert_eq!(
            super::super::super::prepared_tensor_map::native_attempts(),
            before
        );
        assert!(CAPTURE.with(|state| state.borrow().as_ref().unwrap().source_key == "outer"));
        drop(guard);
        assert!(CAPTURE.with(|state| state.borrow().is_none()));
    }
}

/// Genuine grouped-query projection geometry. At most four tokens, at most
/// sixteen attention heads, and ordinary Publisher/Manifest admission throughout.
/// No post-compilation bounds change, alternate kernel, or admission retry.
pub(in crate::replicated_decoder) fn public_projection_geometry_manifest(
    hidden: usize,
    rows: usize,
    reject_prompt: bool,
) -> Result<Manifest, String> {
    // Qwen2 derives head width from hidden/heads. Two-lane heads would require
    // 32/33 heads, exceeding the unchanged ordinary source ceiling of sixteen.
    // Preserve the Q projection width and genuine 128/132/256/264-lane bodies;
    // only factor that width into admissible even rotary heads (KV heads = 1).
    let (heads, head_dim) = match (hidden, rows) {
        (64, 2 | 4) => (16, 4),
        (66, 2 | 4) => (11, 6),
        _ => return Err("unsupported original projection fixture geometry".into()),
    };
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type": "qwen2", "hidden_size": hidden, "intermediate_size": 2,
        "num_hidden_layers": 1, "num_attention_heads": heads,
        "num_key_value_heads": 1, "head_dim": head_dim, "vocab_size": 3,
        "hidden_act": "silu", "tie_word_embeddings": false,
        "attention_bias": true, "max_position_embeddings": 16,
        "rope_theta": 10000, "rms_norm_eps": 0.000001
    }))
    .map_err(|error| error.to_string())?;
    let execution = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        max_abs_activation: 1,
        ..ExecutionPolicy::default()
    };
    let mut publisher = Publisher::new_with_policy(
        &config,
        rows,
        1,
        [1; 32],
        [2; 32],
        WeightMode::Public,
        execution,
    )?;
    if !publisher
        .shapes()
        .iter()
        .any(|(key, _)| key == Q_PROJECTION_SOURCE_KEY)
    {
        return Err("fixture Q projection source absent".into());
    }
    let shapes = publisher.shapes().to_vec();
    for (key, shape) in shapes {
        let bytes = projection_geometry_bytes(&key, &shape, hidden, reject_prompt);
        for (chunk, bytes) in bytes.chunks(4096 * 4).enumerate() {
            publisher.append(&key, chunk * 4096, bytes)?;
        }
    }
    publisher.seal(&[1, 2, 1, 2][..rows])?;
    serde_json::from_slice(&publisher.manifest()?).map_err(|error| error.to_string())
}

fn projection_geometry_bytes(
    key: &str,
    shape: &[usize],
    hidden: usize,
    reject_prompt: bool,
) -> Vec<u8> {
    (0..shape.iter().product::<usize>())
        .flat_map(|index| {
            let value = if key.contains("norm") {
                // Leave numerical headroom below the unchanged activation cap;
                // the native RMS factory still consumes these genuine values.
                0.5f32
            } else if key == "model.embed_tokens.weight" {
                // Both sources require a genuine range guard. Accepted input
                // selects only rows one/two (.25); unused row zero (1.5) widens
                // its authenticated interval beyond the unchanged cap of one.
                // Rejected input selects rows one/two (2), retaining accepted
                // row zero (.25): rejection is private selection, not a
                // statically impossible guard or changed activation budget.
                if reject_prompt && index / hidden != 0 {
                    2f32
                } else if !reject_prompt && index / hidden == 0 {
                    1.5f32
                } else {
                    0.25f32
                }
            } else if key == Q_PROJECTION_SOURCE_KEY {
                1f32 / 512f32
            } else {
                0f32
            };
            value.to_le_bytes()
        })
        .collect()
}

/// Reimport the actual deterministic publisher bytes, through ordinary file
/// authentication. Descriptor inspection chooses fixture data, not authority:
/// importer.finish must still verify every original source commitment.
pub(in crate::replicated_decoder) fn import_projection_geometry_source(
    manifest: &Manifest,
) -> Result<PublicSource, String> {
    let embedding = manifest
        .tensors
        .get("model.embed_tokens.weight")
        .ok_or("projection geometry embedding absent")?;
    let hidden = *embedding.shape.get(1).ok_or("projection embedding shape")?;
    if !matches!(hidden, 64 | 66) {
        return Err("projection import geometry".into());
    }
    let reject_prompt = match embedding.upper {
        upper if upper == ONE * 3 / 2 => false,
        upper if upper == ONE * 2 => true,
        _ => return Err("projection import embedding geometry".into()),
    };
    let mut importer = PublicSourceImport::new(manifest)?;
    for (key, tensor) in &manifest.tensors {
        let bytes = projection_geometry_bytes(key, &tensor.shape, hidden, reject_prompt);
        for (chunk, bytes) in bytes.chunks(4096 * 4).enumerate() {
            importer.append(key, chunk * 4096, bytes)?;
        }
    }
    importer.finish()
}
