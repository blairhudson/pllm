//! Framing/lifecycle tests. Sealed synthetic reply frames below test the codec
//! only, not private execution or authentication. The final test obtains genuine
//! TokenShares through two native Workers and fresh dealer-issued primitive keys.
use super::*;

#[test]
fn shared_phase_geometry_keeps_channel_native_limits_and_combined_extent() {
    let base = manifest(2, 3).geometry;
    for (initial, selections) in [
        (0, 1), (1, 0), (usize::MAX, 2), (1, usize::MAX),
        (30, 2), (32, 2), (64, 8), (16, 2), (14, 4), (1, 5),
    ] {
        assert!(matches!(Manifest::from_geometry(Geometry { initial, selections, ..base }), Err(Error::Domain)));
    }
    for (initial, selections, padded) in [(16, 1, 16), (15, 2, 16), (13, 4, 16), (2, 3, 4)] {
        let geometry = pllm_models::DecoderPlan::validated_phase_geometry(initial, selections).unwrap();
        assert_eq!(geometry.padded_rows(), padded);
        let manifest = manifest(initial, selections);
        assert_eq!(manifest.initial_tokens(), initial);
        assert_eq!(manifest.selections(), selections);
        assert!(manifest.resources().input_frame_bytes <= MAX_INPUT_FRAME_BYTES);
        assert!(manifest.resources().output_frame_bytes <= MAX_OUTPUT_FRAME_BYTES);
    }
}

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
fn decoder_manifest(vocabulary: usize) -> Manifest {
    Manifest::from_geometry(Geometry {
        vocabulary,
        profile: 1,
        ..manifest(2, 3).geometry
    })
    .unwrap()
}
fn checked_manifest(selections: usize) -> Manifest {
    Manifest::from_geometry(Geometry {
        profile: 2,
        selections,
        ..decoder_manifest(151936).geometry
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
    let (magic, kind) = request.manifest.geometry.output_namespace();
    let mut frame = new_frame(magic, request.context(), role, values.len(), kind).unwrap();
    for (chunk, value) in frame[FRAME_HEADER_BYTES..FRAME_HEADER_BYTES + 8 * values.len()]
        .chunks_exact_mut(8)
        .zip(values)
    {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    seal(&mut frame);
    frame
}
fn delivered_client(manifest: &Manifest) -> ClientRequest {
    let mut client = client(manifest);
    drop(client.take_input(0).unwrap());
    drop(client.take_input(1).unwrap());
    client
}

#[test]
fn checked_profile_binds_request_and_prices_exact_extra_output_word() {
    let checked = checked_manifest(4);
    let legacy = Manifest::from_geometry(Geometry {
        profile: 1,
        ..checked.geometry
    })
    .unwrap();
    assert_eq!(
        checked.profile(),
        "fusefss.dense-rope-checked-ranges-ring50-q12.v1"
    );
    assert_eq!(
        Manifest::parse(checked.as_bytes(), checked.contract_digest())
            .unwrap()
            .digest(),
        checked.digest()
    );
    assert_ne!(checked.digest(), legacy.digest());
    let a = checked.resources();
    let b = legacy.resources();
    assert_eq!(a.output_frame_bytes, 112 + 5 * 8 + 32);
    assert_eq!(a.output_frame_bytes, MAX_OUTPUT_FRAME_BYTES);
    assert_eq!(a.output_frame_bytes, b.output_frame_bytes + 8);
    assert_eq!(a.client_peak_bytes, b.client_peak_bytes + 16);
    assert_eq!(
        a.worker_output_export_bytes,
        b.worker_output_export_bytes + 16
    );
    assert_eq!(a.worker_input_import_bytes, b.worker_input_import_bytes);
    let c = client(&checked);
    assert!(c.checked_ranges());
    assert!(!client(&legacy).checked_ranges());
    assert_eq!(&c.manifest_bytes()[84..88], &[50, 2, 0, 0]);
    assert!(matches!(
        Request::parse(c.manifest_bytes(), &legacy),
        Err(Error::Context)
    ));
    assert!(ClientRequest::new(
        checked.contract_digest(),
        [5; 32],
        [checked.as_bytes(), legacy.as_bytes(), checked.as_bytes()],
        &[3, 3]
    )
    .is_err());
    for geometry in [
        Geometry {
            profile: 3,
            ..checked.geometry
        },
        Geometry {
            selections: 5,
            ..checked.geometry
        },
    ] {
        assert!(Manifest::from_geometry(geometry).is_err());
    }
}

#[test]
fn checked_validity_one_reconstructs_only_selected_ids_including_boundary_ids() {
    for selections in 1..=4 {
        let m = checked_manifest(selections);
        let c = delivered_client(&m);
        let expected = [151935u32, 0, 32768, 7];
        let mut left = vec![MASK; selections + 1];
        left[0] = 17;
        let right: Vec<_> = expected[..selections]
            .iter()
            .map(|&id| u64::from(id))
            .chain([1])
            .zip(&left)
            .map(|(value, share)| value.wrapping_sub(*share) & MASK)
            .collect();
        let a = synthetic_reply(&c.request, 0, &left);
        let b = synthetic_reply(&c.request, 1, &right);
        assert_eq!(&a[..8], b"PLLMOC01");
        assert_eq!(&a[104..108], &[0, 50, 2, 0]);
        assert_eq!(&a[108..112], &((selections + 1) as u32).to_le_bytes());
        assert_eq!(a.capacity(), m.resources().output_frame_bytes);
        let tokens = c.finish(&a, &b).unwrap();
        assert_eq!(&*tokens, &expected[..selections]);
        assert_eq!(tokens.capacity(), selections);
    }
}

#[test]
fn checked_zero_validity_rejects_whole_request_before_output_allocation() {
    let m = checked_manifest(4);
    let c = delivered_client(&m);
    // Individual A50 validity shares need not be bits. Every reconstructed
    // token and the final flag is zero, including modular wrap at MASK.
    let a = synthetic_reply(&c.request, 0, &[MASK; 5]);
    let b = synthetic_reply(&c.request, 1, &[1; 5]);
    FAIL_AFTER.with(|n| n.set(0));
    let result = c.finish(&a, &b);
    FAIL_AFTER.with(|n| n.set(-1));
    assert!(matches!(result, Err(Error::Domain)));
}

#[test]
fn checked_nonboolean_or_contradictory_output_is_protocol_failure_not_numeric_rejection() {
    let m = checked_manifest(4);
    for (tokens, validity) in [
        ([0, 0, 0, 0], 2),
        ([0, 0, 0, 0], MASK),
        ([0, 0, 0, 0], 1 << 50), // Noncanonical share, not a reconstructed bit.
        ([1, 0, 0, 0], 0),
        ([0, 0, 0, 1], 0),
        ([0, 0, 0, MASK], 0),
        ([0, 0, 0, 151936], 1),
    ] {
        let c = delivered_client(&m);
        let a = synthetic_reply(&c.request, 0, &[0; 5]);
        let mut words = tokens.to_vec();
        words.push(validity);
        let b = synthetic_reply(&c.request, 1, &words);
        FAIL_AFTER.with(|n| n.set(0));
        let result = c.finish(&a, &b);
        FAIL_AFTER.with(|n| n.set(-1));
        assert!(
            matches!(result, Err(Error::Context)),
            "{tokens:?}, {validity}"
        );
    }
}

#[test]
fn checked_finish_validates_both_complete_frames_before_classifying_zero_validity() {
    let m = checked_manifest(2);
    for index in [0, 8, 40, 72, 104, 105, 106, 107, 108, 112, 136, 167] {
        let c = delivered_client(&m);
        let a = synthetic_reply(&c.request, 0, &[0; 3]);
        let mut b = synthetic_reply(&c.request, 1, &[0; 3]);
        if index == 112 {
            b[index..index + 8].copy_from_slice(&(1u64 << 50).to_le_bytes());
        } else {
            b[index] ^= 1;
        }
        if index < 136 {
            seal(&mut b);
        }
        FAIL_AFTER.with(|n| n.set(0));
        let result = c.finish(&a, &b);
        FAIL_AFTER.with(|n| n.set(-1));
        assert!(matches!(result, Err(Error::Context)), "offset {index}");
    }
    for count in [2, 4, 5] {
        let c = delivered_client(&m);
        let a = synthetic_reply(&c.request, 0, &[0; 3]);
        let b = synthetic_reply(&c.request, 1, &vec![0; count]);
        assert!(matches!(c.finish(&a, &b), Err(Error::Context)));
    }
}

#[test]
fn checked_and_legacy_outputs_never_share_a_wire_namespace() {
    for profile in [1, 2] {
        let m = Manifest::from_geometry(Geometry {
            profile,
            ..checked_manifest(2).geometry
        })
        .unwrap();
        let c = delivered_client(&m);
        let a = synthetic_reply(&c.request, 0, &vec![0; m.geometry.output_words()]);
        let mut b = synthetic_reply(&c.request, 1, &vec![0; m.geometry.output_words()]);
        let (foreign, kind) = if profile == 2 {
            (OUTPUT_MAGIC, 1)
        } else {
            (CHECKED_OUTPUT_MAGIC, 2)
        };
        b[..8].copy_from_slice(foreign);
        b[106] = kind;
        seal(&mut b);
        assert!(matches!(c.finish(&a, &b), Err(Error::Context)));
    }
}

#[test]
fn checked_request_profile_input_substitution_replay_cancel_and_stale_reply_burn() {
    let m = checked_manifest(2);
    let legacy = Manifest::from_geometry(Geometry {
        profile: 1,
        ..m.geometry
    })
    .unwrap();
    let mut c = client(&m);
    let mut receiver = Request::parse(c.manifest_bytes(), &m).unwrap();
    let frame = c.take_input(0).unwrap();
    let mut other = client(&legacy);
    let foreign_input = other.take_input(0).unwrap();
    assert!(matches!(
        receiver.take_input(0, &foreign_input),
        Err(Error::Context)
    ));
    assert!(receiver.is_consumed());
    assert!(matches!(
        receiver.take_input(0, &frame),
        Err(Error::Consumed)
    ));
    let mut receiver = Request::parse(c.manifest_bytes(), &m).unwrap();
    drop(receiver.take_input(0, &frame).unwrap());
    assert!(matches!(
        receiver.take_input(0, &frame),
        Err(Error::Consumed)
    ));
    c.cancel();
    assert!(c.is_consumed());
    assert!(matches!(c.take_input(1), Err(Error::Consumed)));

    let mut c = client(&m);
    drop(c.take_input(0).unwrap());
    let a = synthetic_reply(&c.request, 0, &[0; 3]);
    let b = synthetic_reply(&c.request, 1, &[0; 3]);
    assert!(matches!(c.finish(&a, &b), Err(Error::Context))); // Partial delivery is not range failure.

    let stale = Request::parse(&request_bytes(&m, [6; 32]).unwrap(), &m).unwrap();
    let c = delivered_client(&m);
    let a = synthetic_reply(&c.request, 0, &[0; 3]);
    let b = synthetic_reply(&stale, 1, &[0; 3]);
    assert!(matches!(c.finish(&a, &b), Err(Error::Context)));

    let c = delivered_client(&m);
    let a = synthetic_reply(&c.request, 0, &[0; 3]);
    let b = synthetic_reply(&c.request, 1, &[0, 0, 1]);
    FAIL_AFTER.with(|n| n.set(0));
    let result = c.finish(&a, &b);
    FAIL_AFTER.with(|n| n.set(-1));
    assert!(matches!(result, Err(Error::Allocation)));
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
fn capsule_manifest_retains_exact_legacy_bytes() {
    let expected = concat!(
        r#"{"contract":"0101010101010101010101010101010101010101010101010101010101010101","#,
        r#""initial":2,"input_digest":"0202020202020202020202020202020202020202020202020202020202020202","#,
        r#""profile":"fusefss.gpt2-complete-ring50-q12.v1","#,
        r#""program":"0303030303030303030303030303030303030303030303030303030303030303","#,
        r#""schema":"pllm.fss.channel.program.v1","selections":2,"#,
        r#""source":"0404040404040404040404040404040404040404040404040404040404040404","vocabulary":7}"#,
    );
    let m = manifest(2, 2);
    assert_eq!(m.as_bytes(), expected.as_bytes());
    assert_eq!(
        Manifest::parse(expected.as_bytes(), [1; 32])
            .unwrap()
            .digest(),
        m.digest()
    );
}

#[test]
fn both_legacy_profiles_keep_request_and_output_wire_images() {
    // Independent stdlib SHA-256/struct.pack construction of the pre-D14 wire
    // schema, with the same public fixture geometry as manifest(2, 2).
    for (profile, request_hash, output_hash) in [
        (
            0,
            "607d2c38dbe5194a35413dfcd507add61d849e7887e5bfacf3aa273f028676e6",
            "ecf929bd5d6cbc7c99a163fb7d818a82c45601f1204d3e27c93620b51696fddd",
        ),
        (
            1,
            "91cf32eacb368a2780e7c996abdad9e4eaa0b2bb83944312119bf881ed6211ab",
            "cd551b683f3c611f9dc17b15bfcbe0444fccce88cee0e066a9e4f94722d566d4",
        ),
    ] {
        let m = Manifest::from_geometry(Geometry {
            profile,
            ..manifest(2, 2).geometry
        })
        .unwrap();
        let c = client(&m);
        let output = synthetic_reply(&c.request, 0, &[MASK, 4]);
        assert_eq!(c.manifest_bytes().len(), 248);
        assert_eq!(output.len(), 160);
        assert_eq!(
            format!("{:x}", Sha256::digest(c.manifest_bytes())),
            request_hash
        );
        assert_eq!(format!("{:x}", Sha256::digest(&output)), output_hash);
    }
}

#[test]
fn decoder_vocabulary_boundary_ids_round_trip_with_fixed_frame_storage() {
    let small = manifest(2, 3).resources();
    for vocabulary in [32769, 151936, 262144] {
        let m = decoder_manifest(vocabulary);
        assert_eq!(
            Manifest::parse(m.as_bytes(), [1; 32]).unwrap().vocabulary(),
            vocabulary
        );
        let resources = m.resources();
        assert_eq!(resources.client_peak_bytes, small.client_peak_bytes);
        assert_eq!(
            resources.worker_input_import_bytes,
            small.worker_input_import_bytes
        );
        assert_eq!(
            resources.worker_output_export_bytes,
            small.worker_output_export_bytes
        );

        let ids = [0, vocabulary as u32 - 1];
        let mut c = ClientRequest::new([1; 32], [5; 32], [m.as_bytes(); 3], &ids).unwrap();
        assert_eq!(c.manifest_bytes().len(), REQUEST_BYTES);
        assert_eq!(
            &c.manifest_bytes()[80..84],
            &(vocabulary as u32).to_le_bytes()
        );
        let mut r0 = Request::parse(c.manifest_bytes(), &m).unwrap();
        let mut r1 = Request::parse(c.manifest_bytes(), &m).unwrap();
        let f0 = c.take_input(0).unwrap();
        let f1 = c.take_input(1).unwrap();
        assert_eq!(f0.len(), small.input_frame_bytes);
        assert_eq!(f1.len(), small.input_frame_bytes);
        let x0 = r0.take_input(0, &f0).unwrap();
        let x1 = r1.take_input(1, &f1).unwrap();
        assert_eq!(x0.capacity(), ids.len());
        assert_eq!(x1.capacity(), ids.len());
        for i in 0..ids.len() {
            assert_eq!(x0[i].wrapping_add(x1[i]) & MASK, u64::from(ids[i]));
        }

        // Codec-only replies: exercise wrapping shares and the final ID boundary.
        let expected = [vocabulary as u32 - 1, 32768, 0];
        let left = [MASK, 17, MASK - 5];
        let right: Vec<_> = expected
            .iter()
            .zip(left)
            .map(|(&id, share)| u64::from(id).wrapping_sub(share) & MASK)
            .collect();
        let a = synthetic_reply(&r0, 0, &left);
        let b = synthetic_reply(&r1, 1, &right);
        assert_eq!(a.len(), small.output_frame_bytes);
        assert_eq!(b.len(), small.output_frame_bytes);
        let actual = c.finish(&a, &b).unwrap();
        assert_eq!(&*actual, &expected);
        assert_eq!(actual.capacity(), expected.len());
    }
}

#[test]
fn decoder_codec_rejects_excess_vocabulary_and_invalid_ids_before_allocation() {
    let m = decoder_manifest(151936);
    for vocabulary in [0, 262145, usize::MAX] {
        assert!(Manifest::from_geometry(Geometry {
            vocabulary,
            ..m.geometry
        })
        .is_err());
    }
    let canonical = std::str::from_utf8(m.as_bytes()).unwrap();
    let mut draws = 0;
    for value in [
        "0",
        "262145",
        "999999",
        "18446744073709551615",
        "-1",
        "0151936",
        "151936.0",
    ] {
        let malformed =
            canonical.replace("\"vocabulary\":151936", &format!("\"vocabulary\":{value}"));
        FAIL_AFTER.with(|n| n.set(0));
        let result = ClientRequest::with_random(
            [1; 32],
            [5; 32],
            [malformed.as_bytes(); 3],
            &[0, 151935],
            |_| {
                draws += 1;
                Ok(())
            },
        );
        FAIL_AFTER.with(|n| n.set(-1));
        assert!(
            matches!(result, Err(Error::Context | Error::Domain)),
            "{value}"
        );
    }
    for invalid in [151936, u32::MAX] {
        FAIL_AFTER.with(|n| n.set(0));
        let result =
            ClientRequest::with_random([1; 32], [5; 32], [m.as_bytes(); 3], &[0, invalid], |_| {
                draws += 1;
                Ok(())
            });
        FAIL_AFTER.with(|n| n.set(-1));
        assert!(matches!(result, Err(Error::Domain)));
    }
    assert_eq!(draws, 0);

    let mut c = client(&m);
    let request = Request::parse(c.manifest_bytes(), &m).unwrap();
    drop(c.take_input(0).unwrap());
    drop(c.take_input(1).unwrap());
    // First two IDs are 0 and V-1; only the final reconstructed ID equals V.
    let a = synthetic_reply(&request, 0, &[MASK, 7, MASK]);
    let b = synthetic_reply(&request, 1, &[1, 151928, 151937]);
    FAIL_AFTER.with(|n| n.set(0));
    let result = c.finish(&a, &b);
    FAIL_AFTER.with(|n| n.set(-1));
    assert!(matches!(result, Err(Error::Domain)));
}

#[test]
fn decoder_frames_reject_stale_sessions_and_substituted_source_bindings() {
    let m = decoder_manifest(151936);
    for changed in 0..6 {
        let mut geometry = m.geometry;
        match changed {
            1 => geometry.source[0] ^= 1,
            2 => geometry.input[0] ^= 1,
            3 => geometry.program[0] ^= 1,
            4 => geometry.vocabulary -= 1,
            5 => geometry.contract[0] ^= 1,
            _ => {}
        }
        let foreign = Manifest::from_geometry(geometry).unwrap();
        let session = if changed == 0 { [6; 32] } else { [5; 32] };
        let foreign_request =
            Request::parse(&request_bytes(&foreign, session).unwrap(), &foreign).unwrap();
        let mut c = client(&m);
        if changed != 0 {
            assert!(matches!(
                Request::parse(c.manifest_bytes(), &foreign),
                Err(Error::Context)
            ));
            assert!(ClientRequest::new(
                [1; 32],
                [5; 32],
                [m.as_bytes(), foreign.as_bytes(), m.as_bytes()],
                &[0, 151935],
            )
            .is_err());
        }
        let mut local = Request::parse(c.manifest_bytes(), &m).unwrap();
        let valid_input = c.take_input(0).unwrap();
        drop(c.take_input(1).unwrap());
        let mut stale_input = new_frame(INPUT_MAGIC, foreign_request.context(), 0, 2, 0).unwrap();
        seal(&mut stale_input);
        FAIL_AFTER.with(|n| n.set(0));
        let result = local.take_input(0, &stale_input);
        FAIL_AFTER.with(|n| n.set(-1));
        assert!(matches!(result, Err(Error::Context)));
        assert!(local.is_consumed());
        assert!(matches!(
            local.take_input(0, &valid_input),
            Err(Error::Consumed)
        ));

        let a = synthetic_reply(&c.request, 0, &[0, 0, 0]);
        let b = synthetic_reply(&foreign_request, 1, &[0, 0, 0]);
        FAIL_AFTER.with(|n| n.set(0));
        let result = c.finish(&a, &b);
        FAIL_AFTER.with(|n| n.set(-1));
        assert!(matches!(result, Err(Error::Context)));
    }
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
    let mut program = tiny_program();
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

    // Even an invalid test-only Program cannot export beyond its Capsule policy.
    program.vocabulary = Capacity::Capsule.max_vocabulary() + 1;
    assert!(matches!(
        Manifest::new([1; 32], [2; 32], &program),
        Err(Error::Domain)
    ));
}
