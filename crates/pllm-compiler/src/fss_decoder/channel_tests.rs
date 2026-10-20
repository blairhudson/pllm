//! Framing/lifecycle tests. Sealed synthetic reply frames below test the codec
//! only, not private execution or authentication. The final test obtains genuine
//! TokenShares through two native Workers and fresh dealer-issued primitive keys.
use super::*;

fn manifest(initial: usize, selections: usize) -> Manifest {
    Manifest::from_geometry(Geometry {
        contract: [1; 32],
        input: [2; 32],
        program: [3; 32],
        source: [4; 32],
        profile: 0,
        initial,
        selections,
        vocabulary: 7,
    })
    .unwrap()
}
fn client(m: &Manifest) -> ClientRequest {
    ClientRequest::new(
        m.contract_digest(),
        [5; 32],
        [m.as_bytes(); 3],
        &vec![3; m.initial_tokens()],
    )
    .unwrap()
}
fn synthetic_reply(request: &Request, role: u8, values: &[u64]) -> Zeroizing<Vec<u8>> {
    let mut frame = new_frame(OUTPUT_MAGIC, request.context(), role, values.len(), 1).unwrap();
    for (chunk, value) in frame[FRAME_HEADER_BYTES..FRAME_HEADER_BYTES + 8 * values.len()]
        .chunks_exact_mut(8)
        .zip(values)
    {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    seal(&mut frame);
    frame
}

#[test]
fn canonical_manifest_and_request_bind_original_and_coded_sources() {
    let m = manifest(2, 2);
    let value: serde_json::Value = serde_json::from_slice(m.as_bytes()).unwrap();
    assert_eq!(pllm_types::canonical_bytes(&value), m.as_bytes());
    assert_eq!(
        Manifest::parse(m.as_bytes(), [1; 32]).unwrap().as_bytes(),
        m.as_bytes()
    );
    let c = client(&m);
    assert_eq!(c.manifest_bytes().len(), REQUEST_BYTES);
    let request = Request::parse(c.manifest_bytes(), &m).unwrap();
    let context = request.context();
    assert_eq!(context.plan, m.program_digest());
    assert_eq!(context.session, [5; 32]);
    assert_eq!(context.first_tensor_index, 0);
    assert_ne!(context.operation, [0; 32]);
    for changed in 0..8 {
        let mut g = m.geometry;
        match changed {
            0 => g.contract[0] ^= 1,
            1 => g.input[0] ^= 1,
            2 => g.source[0] ^= 1,
            3 => g.program[0] ^= 1,
            4 => g.profile = 1,
            5 => g.initial += 1,
            6 => g.selections += 1,
            _ => g.vocabulary += 1,
        }
        let foreign = Manifest::from_geometry(g).unwrap();
        assert!(matches!(
            Request::parse(c.manifest_bytes(), &foreign),
            Err(Error::Context)
        ));
        assert!(ClientRequest::new(
            [1; 32],
            [5; 32],
            [m.as_bytes(), foreign.as_bytes(), m.as_bytes()],
            &[3, 3]
        )
        .is_err());
    }
    let mut g = m.geometry;
    g.input = [0; 32];
    assert!(Manifest::from_geometry(g).is_err());
}

#[test]
fn validation_and_both_allocations_precede_randomness() {
    let m = manifest(2, 2);
    let mut draws = 0;
    for (contract, session, ids) in [
        ([0; 32], [5; 32], vec![1, 2]),
        ([1; 32], [0; 32], vec![1, 2]),
        ([1; 32], [5; 32], vec![1]),
        ([1; 32], [5; 32], vec![1, 7]),
        ([1; 32], [5; 32], vec![u32::MAX, 0]),
    ] {
        assert!(
            ClientRequest::with_random(contract, session, [m.as_bytes(); 3], &ids, |_| {
                draws += 1;
                Ok(())
            })
            .is_err()
        );
    }
    assert_eq!(draws, 0);
    for fail in 0..2 {
        FAIL_AFTER.with(|n| n.set(fail));
        let result =
            ClientRequest::with_random([1; 32], [5; 32], [m.as_bytes(); 3], &[1, 2], |_| {
                draws += 1;
                Ok(())
            });
        FAIL_AFTER.with(|n| n.set(-1));
        assert!(matches!(result, Err(Error::Allocation)));
    }
    assert_eq!(draws, 0);
    for fail in 0..2 {
        let mut calls = 0;
        let result =
            ClientRequest::with_random([1; 32], [5; 32], [m.as_bytes(); 3], &[1, 2], |bytes| {
                if calls == fail {
                    return Err(Error::Randomness);
                }
                calls += 1;
                bytes.fill(0xff);
                Ok(())
            });
        assert!(matches!(result, Err(Error::Randomness)));
    }
}

#[test]
fn fresh_input_frames_reconstruct_only_at_client_test_boundary_and_have_exact_capacities() {
    let m = manifest(3, 2);
    let mut a = ClientRequest::new([1; 32], [5; 32], [m.as_bytes(); 3], &[0, 3, 6]).unwrap();
    let mut b = ClientRequest::new([1; 32], [5; 32], [m.as_bytes(); 3], &[0, 3, 6]).unwrap();
    let mut r0 = Request::parse(a.manifest_bytes(), &m).unwrap();
    let mut r1 = Request::parse(a.manifest_bytes(), &m).unwrap();
    let f0 = a.take_input(0).unwrap();
    let f1 = a.take_input(1).unwrap();
    let again = b.take_input(0).unwrap();
    assert_eq!(f0.capacity(), f0.len());
    assert_eq!(f1.capacity(), f1.len());
    assert_ne!(
        &f0[FRAME_HEADER_BYTES..f0.len() - 32],
        &again[FRAME_HEADER_BYTES..again.len() - 32]
    );
    let x0 = r0.take_input(0, &f0).unwrap();
    let x1 = r1.take_input(1, &f1).unwrap();
    assert_eq!(x0.capacity(), 3);
    assert_eq!(x1.capacity(), 3);
    for (i, expected) in [0, 3, 6].into_iter().enumerate() {
        assert_eq!(x0[i].wrapping_add(x1[i]) & MASK, expected);
    }
    assert!(matches!(r0.take_input(0, &f0), Err(Error::Consumed)));
    assert!(r0.is_consumed());
    assert!(matches!(a.take_input(0), Err(Error::Consumed)));
    assert!(a.is_consumed());
}

#[test]
fn partial_delivery_cancel_wrong_role_and_early_finish_burn() {
    let m = manifest(1, 1);
    let mut c = client(&m);
    let f = c.take_input(0).unwrap();
    c.cancel();
    assert!(c.is_consumed());
    assert!(c.take_input(1).is_err());
    assert_eq!(f.capacity(), f.len()); // Exported recipient owner outlives cancellation.
    let mut c = client(&m);
    assert!(c.take_input(2).is_err());
    assert!(c.is_consumed());
    let c = client(&m);
    let r = Request::parse(c.manifest_bytes(), &m).unwrap();
    let a = synthetic_reply(&r, 0, &[1]);
    let b = synthetic_reply(&r, 1, &[1]);
    assert!(matches!(c.finish(&a, &b), Err(Error::Context)));
    let mut c = client(&m);
    let frame = c.take_input(0).unwrap();
    let mut r = Request::parse(c.manifest_bytes(), &m).unwrap();
    assert!(matches!(r.take_input(1, &frame), Err(Error::Context)));
    assert!(r.is_consumed());
    assert!(matches!(r.take_input(0, &frame), Err(Error::Consumed)));
}

#[test]
fn strict_manifest_request_and_frame_preflight_reject_without_allocating() {
    let m = manifest(1, 1);
    let mut c = client(&m);
    let frame = c.take_input(0).unwrap();
    for index in [0, 8, 40, 72, 104, 105, 106, 107, 108, 112, frame.len() - 1] {
        let mut corrupted = Zeroizing::new(frame.to_vec());
        corrupted[index] ^= 1;
        if index < frame.len() - 32 {
            seal(&mut corrupted);
        }
        let mut r = Request::parse(c.manifest_bytes(), &m).unwrap();
        FAIL_AFTER.with(|n| n.set(0));
        let result = r.take_input(0, &corrupted);
        FAIL_AFTER.with(|n| n.set(-1));
        // The body mutation may still be a canonical ring share; checksums do
        // not authenticate a malicious sender. Force a noncanonical body below.
        if index != 112 {
            assert!(matches!(result, Err(Error::Context)), "offset {index}");
        } else {
            assert!(matches!(result, Err(Error::Allocation)));
        }
        assert!(r.is_consumed());
    }
    let mut corrupted = Zeroizing::new(frame.to_vec());
    corrupted[FRAME_HEADER_BYTES..FRAME_HEADER_BYTES + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    seal(&mut corrupted);
    let mut r = Request::parse(c.manifest_bytes(), &m).unwrap();
    FAIL_AFTER.with(|n| n.set(0));
    assert!(matches!(r.take_input(0, &corrupted), Err(Error::Context)));
    FAIL_AFTER.with(|n| n.set(-1));
    for size in [0, 1, REQUEST_BYTES - 1, REQUEST_BYTES + 1] {
        assert!(Request::parse(&vec![0; size], &m).is_err());
    }
    for index in [
        0, 8, 40, 72, 76, 80, 84, 85, 86, 88, 120, 152, 184, 216, 247,
    ] {
        let mut bytes = c.manifest_bytes().to_vec();
        bytes[index] ^= 1;
        assert!(Request::parse(&bytes, &m).is_err());
    }
    for malformed in [
        b"{}".as_slice(),
        b"[]",
        b"null",
        b"{\"initial\":99999999999999999999}",
    ] {
        assert!(Manifest::parse(malformed, [1; 32]).is_err());
    }
    let mut noncanonical = m.as_bytes().to_vec();
    noncanonical.push(b' ');
    assert!(Manifest::parse(&noncanonical, [1; 32]).is_err());
    assert!(matches!(
        Manifest::parse(&vec![b' '; MAX_MANIFEST_BYTES + 1], [1; 32]),
        Err(Error::Resource)
    ));
}

#[test]
fn final_only_reconstruction_checks_context_role_ring_and_every_token_before_allocation() {
    let m = manifest(1, 2);
    let mut c = client(&m);
    let request = Request::parse(c.manifest_bytes(), &m).unwrap();
    drop(c.take_input(0).unwrap());
    drop(c.take_input(1).unwrap());
    let a = synthetic_reply(&request, 0, &[MASK, 4]);
    let b = synthetic_reply(&request, 1, &[2, 2]);
    let out = c.finish(&a, &b).unwrap();
    assert_eq!(&*out, &[1, 6]);
    assert_eq!(out.capacity(), 2);
    for invalid in 0..5 {
        let mut c = client(&m);
        let r = Request::parse(c.manifest_bytes(), &m).unwrap();
        drop(c.take_input(0).unwrap());
        drop(c.take_input(1).unwrap());
        let mut a = synthetic_reply(&r, 0, &[1, 1]);
        let mut b = synthetic_reply(&r, 1, &[1, 1]);
        match invalid {
            0 => b[104] = 0,
            1 => b[8] ^= 1,
            2 => b[FRAME_HEADER_BYTES..FRAME_HEADER_BYTES + 8]
                .copy_from_slice(&u64::MAX.to_le_bytes()),
            3 => b[FRAME_HEADER_BYTES + 8..FRAME_HEADER_BYTES + 16]
                .copy_from_slice(&6u64.to_le_bytes()),
            _ => a[..8].copy_from_slice(INPUT_MAGIC),
        }
        seal(&mut a);
        seal(&mut b);
        FAIL_AFTER.with(|n| n.set(0));
        let result = c.finish(&a, &b);
        FAIL_AFTER.with(|n| n.set(-1));
        if invalid == 3 {
            assert!(matches!(result, Err(Error::Domain)));
        } else {
            assert!(matches!(result, Err(Error::Context)));
        }
    }
}

#[test]
fn fixed_resource_bounds_and_allocation_failures_leave_no_reusable_owner() {
    let m = manifest(16, 1);
    let r = m.resources();
    assert_eq!(r.input_frame_bytes, MAX_INPUT_FRAME_BYTES);
    assert!(r.output_frame_bytes <= MAX_OUTPUT_FRAME_BYTES);
    assert_eq!(r.request_bytes, REQUEST_BYTES);
    assert!(r.manifest_bytes <= MAX_MANIFEST_BYTES);
    assert!(r.client_peak_bytes < 16 << 10);
    assert!(r.worker_input_import_bytes < 16 << 10);
    assert!(r.worker_output_export_bytes < 16 << 10);
    assert!(matches!(
        allocate::<u8>(super::super::MAX_BYTES + 1),
        Err(Error::Resource)
    ));
    assert!(new_frame(INPUT_MAGIC, client(&m).request.context(), 0, 17, 0).is_err());
    assert!(new_frame(OUTPUT_MAGIC, client(&m).request.context(), 0, 5, 1).is_err());
    assert!(Manifest::from_geometry(Geometry {
        initial: 16,
        selections: 2,
        ..m.geometry
    })
    .is_err());
    assert!(Manifest::from_geometry(Geometry {
        initial: 17,
        ..m.geometry
    })
    .is_err());
    assert!(Manifest::from_geometry(Geometry {
        selections: 5,
        ..manifest(1, 1).geometry
    })
    .is_err());
    let m = manifest(1, 1);
    let mut c = client(&m);
    let f = c.take_input(0).unwrap();
    let mut request = Request::parse(c.manifest_bytes(), &m).unwrap();
    FAIL_AFTER.with(|n| n.set(0));
    assert!(matches!(request.take_input(0, &f), Err(Error::Allocation)));
    FAIL_AFTER.with(|n| n.set(-1));
    assert!(request.is_consumed());
    assert!(request.take_input(0, &f).is_err());
    let mut c = client(&m);
    let request = Request::parse(c.manifest_bytes(), &m).unwrap();
    drop(c.take_input(0).unwrap());
    drop(c.take_input(1).unwrap());
    let a = synthetic_reply(&request, 0, &[0]);
    let b = synthetic_reply(&request, 1, &[1]);
    FAIL_AFTER.with(|n| n.set(0));
    assert!(matches!(c.finish(&a, &b), Err(Error::Allocation)));
    FAIL_AFTER.with(|n| n.set(-1));
}

// Generated F3-size source only. No external checkpoint, model launcher, clear
// operator oracle or fabricated NativeOutput/TokenShares is involved.
fn tiny_program() -> Program {
    use crate::fss_decoder::{Source, Tensor};
    use pllm_models::{DecoderWorkload, ModelOperator as Op};
    let config = serde_json::json!({"model_type":"gpt2","n_embd":2,"n_head":1,"n_layer":1,
        "n_positions":2,"vocab_size":2,"n_inner":2});
    let plan = pllm_models::lower_model_json(
        &serde_json::to_vec(&config).unwrap(),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1,
            max_new_tokens: 1,
        },
    )
    .unwrap();
    let mut tensors = Vec::<Tensor>::new();
    for op in &plan.prefill.operations {
        let Some(id) = op.attributes["weight"].as_str() else {
            continue;
        };
        if tensors.iter().any(|t| t.id() == id) {
            continue;
        }
        match op.operator {
            Op::TokenLookup => {
                let codes = if op.inputs[0] == "input.tokens" {
                    [4096, 0, 0, 4096]
                } else {
                    [0; 4]
                };
                tensors.push(Tensor::from_q12(id, &[2, 2], &codes).unwrap());
            }
            Op::LayerNorm => {
                tensors.push(Tensor::from_q12(id, &[2], &[4096, 4096]).unwrap());
                tensors.push(
                    Tensor::from_q12(op.attributes["bias"].as_str().unwrap(), &[2], &[0, 0])
                        .unwrap(),
                );
            }
            Op::Linear => {
                let shape: Vec<usize> = op.attributes["weight_shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap() as usize)
                    .collect();
                tensors
                    .push(Tensor::from_q12(id, &shape, &vec![0; shape.iter().product()]).unwrap());
                tensors.push(
                    Tensor::from_q12(
                        op.attributes["bias"].as_str().unwrap(),
                        &[shape[1]],
                        &vec![0; shape[1]],
                    )
                    .unwrap(),
                );
            }
            _ => panic!("unexpected source artifact"),
        }
    }
    Program::compile(&plan, Source::new(tensors).unwrap(), 1, Limits::default()).unwrap()
}

struct KernelRequest {
    context: Context,
    digest: [u8; 32],
    input: Zeroizing<Vec<u64>>,
    reply: std::sync::mpsc::SyncSender<Result<super::super::NativeOutput, Error>>,
}
struct PairDriver {
    sender: Option<std::sync::mpsc::SyncSender<KernelRequest>>,
    receiver: Option<std::sync::mpsc::Receiver<KernelRequest>>,
    cancelled: bool,
}
fn real_native_pair(
    plan: &pllm_garble::fusefss_backend::composition::Plan<'_>,
    context: Context,
    left: &[u64],
    right: &[u64],
) -> Result<[super::super::NativeOutput; 2], Error> {
    use pllm_garble::fusefss_backend::composition::Event;
    std::thread::scope(|scope| {
        let (asks, rx) = std::sync::mpsc::sync_channel::<usize>(0);
        let (materials, received) = std::sync::mpsc::sync_channel(0);
        let dealer = scope.spawn(move || {
            while let Ok(pc) = rx.recv() {
                if materials.send(plan.issue_step(pc, context)).is_err() {
                    break;
                }
            }
        });
        let result = (|| {
            let mut a = plan.runtime(0, context, left)?;
            let mut b = plan.runtime(1, context, right)?;
            let mut ea = a.poll()?;
            let mut eb = b.poll()?;
            loop {
                match (ea, eb) {
                    (Event::NeedMaterial { pc: a_pc }, Event::NeedMaterial { pc: b_pc }) => {
                        if a_pc != b_pc {
                            return Err(Error::Context);
                        }
                        asks.send(a_pc).map_err(|_| Error::Context)?;
                        let [ma, mb] = received.recv().map_err(|_| Error::Context)??;
                        ea = a.feed(ma)?;
                        eb = b.feed(mb)?;
                    }
                    (Event::Frame(fa), Event::Frame(fb)) => {
                        ea = a.advance(fb.as_bytes())?;
                        eb = b.advance(fa.as_bytes())?;
                    }
                    (Event::Output(oa), Event::Output(ob)) => {
                        return Ok([
                            super::super::NativeOutput::from_native(oa)?,
                            super::super::NativeOutput::from_native(ob)?,
                        ])
                    }
                    _ => return Err(Error::Context),
                }
            }
        })();
        drop(asks);
        drop(received);
        dealer.join().map_err(|_| Error::Context)?;
        result
    })
}
impl super::super::Driver for PairDriver {
    fn run(
        &mut self,
        context: Context,
        plan: &pllm_garble::fusefss_backend::composition::Plan<'_>,
        input: &[u64],
    ) -> Result<super::super::NativeOutput, Error> {
        if let Some(sender) = &self.sender {
            let (reply, received) = std::sync::mpsc::sync_channel(0);
            sender
                .send(KernelRequest {
                    context,
                    digest: plan.digest(),
                    input: Zeroizing::new(super::super::copied(input)?),
                    reply,
                })
                .map_err(|_| Error::Context)?;
            received.recv().map_err(|_| Error::Context)?
        } else {
            let request = self
                .receiver
                .as_ref()
                .ok_or(Error::Context)?
                .recv()
                .map_err(|_| Error::Context)?;
            if request.digest != plan.digest()
                || request.context.plan != context.plan
                || request.context.session != context.session
                || request.context.operation != context.operation
                || request.context.first_tensor_index != context.first_tensor_index
            {
                let _ = request.reply.send(Err(Error::Context));
                return Err(Error::Context);
            }
            match real_native_pair(plan, context, input, &request.input) {
                Ok([a, b]) => {
                    request.reply.send(Ok(b)).map_err(|_| Error::Context)?;
                    Ok(a)
                }
                Err(error) => {
                    let _ = request.reply.send(Err(error));
                    Err(error)
                }
            }
        }
    }
    fn cancel(&mut self) {
        self.cancelled = true;
        self.sender = None;
        self.receiver = None;
    }
}

#[test]
fn genuine_workers_produce_only_bound_final_token_frames() {
    use super::super::Worker;
    let program = tiny_program();
    let manifest = Manifest::new([1; 32], [2; 32], &program).unwrap();
    assert_eq!(manifest.program_digest(), program.digest());
    assert_eq!(manifest.source_digest(), program.source_digest());
    assert!(Manifest::new([1; 32], [0; 32], &program).is_err());
    let mut client = ClientRequest::new([1; 32], [5; 32], [manifest.as_bytes(); 3], &[0]).unwrap();
    let mut request0 = Request::parse(client.manifest_bytes(), &manifest).unwrap();
    let mut request1 = Request::parse(client.manifest_bytes(), &manifest).unwrap();
    let frame0 = client.take_input(0).unwrap();
    let frame1 = client.take_input(1).unwrap();
    let input0 = request0.take_input(0, &frame0).unwrap();
    let input1 = request1.take_input(1, &frame1).unwrap();
    let context = request0.context();
    let (sender, receiver) = std::sync::mpsc::sync_channel(0);
    let (output0, output1) = std::thread::scope(|scope| {
        let a = scope.spawn(|| {
            let mut driver = PairDriver {
                sender: None,
                receiver: Some(receiver),
                cancelled: false,
            };
            let shares = Worker::new(&program, 0, context, input0)
                .unwrap()
                .execute(&mut driver)
                .unwrap();
            assert!(!driver.cancelled);
            shares
        });
        let b = scope.spawn(|| {
            let mut driver = PairDriver {
                sender: Some(sender),
                receiver: None,
                cancelled: false,
            };
            let shares = Worker::new(&program, 1, context, input1)
                .unwrap()
                .execute(&mut driver)
                .unwrap();
            assert!(!driver.cancelled);
            shares
        });
        (a.join().unwrap(), b.join().unwrap())
    });
    let reply0 = request0.output(0, output0).unwrap();
    let reply1 = request1.output(1, output1).unwrap();
    assert!(request0.is_consumed());
    assert!(request1.is_consumed());
    assert_eq!(reply0.capacity(), manifest.resources().output_frame_bytes);
    assert_eq!(&*client.finish(&reply0, &reply1).unwrap(), &[0]);
}
