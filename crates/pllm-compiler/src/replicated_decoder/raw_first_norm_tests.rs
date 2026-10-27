//! Source-only definitions, all UNRUN. Existing synthetic genuine owner only;
//! no exported/generated literal fixture, checkpoint read or source constructor.
use super::*;
use raw_clear_producer::raw_first_norm as norm;

#[test]
fn original_lookup_norm_geometry_quote_refuses_without_factory() {
    let before=norm::OUTPUT_FACTORIES.with(|n|n.get());
    for shape in [[0,1,2],[1,1,0],[usize::MAX,2,2],[1,1,16_777_217]] {
        assert!(norm::demand_for_tests(shape).is_err());
    }
    assert_eq!(norm::OUTPUT_FACTORIES.with(|n|n.get()),before);
}

#[test]
fn original_lookup_norm_continuity_exact_bits() {
    for graph in [RawGraph::Prefill,RawGraph::Decode] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let lookup=genuine_norm_input(&host,graph);
        let root=Arc::downgrade(&host.root);
        let output=lookup.first_bf16_norm().unwrap();
        assert_eq!(output.shape(),match graph {RawGraph::Prefill=>[1,4,2],RawGraph::Decode=>[1,1,2]});
        // Independently fixed BF16 RN-even words, not candidate-derived output.
        // [1,2] and [4,8] cast to [81/128,81/64]; [.5,.25]
        // casts to [81/64,81/128]. Shared direct gamma is exactly [1,2].
        // Epsilon changes RN32 intermediates but not these BF16 cells.
        let expected:&[u16]=match graph {
            RawGraph::Prefill=>&[0x3f22,0x4022,0x3f22,0x4022,0x3fa2,0x3fa2,0x3f22,0x4022],
            RawGraph::Decode=>&[0x3fa2,0x3fa2],
        };
        assert_eq!(output.words(),expected);
        output.check_source(&host).unwrap();
        let foreign=Fixture::new().collect(19).unwrap(); foreign.finish().unwrap();
        assert!(output.check_source(&foreign).is_err());
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
        assert!(host.claim_reader().is_err());
        drop(host);
        let host=RawHost {root:root.upgrade().unwrap()};
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
        drop(output);
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
        host.retire().unwrap(); host.release_final_charges().unwrap(); drop(host);
        assert!(root.upgrade().is_none());
    }
}

#[test]
fn original_lookup_norm_cumulative_exact_work_short_and_sibling_burn() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        // The existing original-source ledger consumes work before lookup.
        // No ledger creation, reset, reissue or decrement occurs here.
        host.inspect(|r|r.header.original.prepay_primitive_work(&r.header.controller,7).unwrap());
        let lookup=genuine_norm_input(&host,RawGraph::Prefill);
        let quote=norm::quote_for_tests(&lookup);
        assert!(quote.work>4*(33792*2+24576));
        let sibling=leave_primitive_work(&host,quote.work,short);
        let before=(norm_counts(),norm::OUTPUT_FACTORIES.with(|n|n.get()));
        let result=lookup.first_bf16_norm();
        assert_eq!(result.is_err(),short);
        if short { assert_eq!((norm_counts(),norm::OUTPUT_FACTORIES.with(|n|n.get())),before); }
        host.inspect(|r| {
            assert_eq!(sibling.prepay_primitive_work(&r.header.controller,0).is_err(),short);
            if !short { assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err()); }
        });
        drop(result);
        assert!(host.claim_reader().is_err());
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    }
}

#[test]
fn original_lookup_norm_exact_bytes_short_before_gamma_or_output() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let lookup=genuine_norm_input(&host,RawGraph::Prefill);
        let quote=norm::quote_for_tests(&lookup);
        let sibling=leave_primitive_work(&host,quote.work,false);
        let mut outside=host.inspect(|r|r.header.root.charge_external(0).unwrap());
        let used=outside.budget().used_bytes();
        outside.resize_payload(MAX_PROGRAM_BYTES-used-quote.resident_bytes-size_of::<OwnedExternalCharge>()+usize::from(short)).unwrap();
        let before=(norm_counts(),norm::OUTPUT_FACTORIES.with(|n|n.get()));
        let result=lookup.first_bf16_norm();
        assert_eq!(result.is_err(),short);
        if short { assert_eq!((norm_counts(),norm::OUTPUT_FACTORIES.with(|n|n.get())),before); }
        else { assert_eq!(outside.budget().used_bytes(),MAX_PROGRAM_BYTES); }
        host.inspect(|r| {
            sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
            assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());
        });
        drop(result); drop(outside);
    }
}

#[test]
fn original_lookup_norm_profile_plan_gamma_epsilon_width_mismatch() {
    for fault in 0..6 {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let lookup=genuine_norm_input(&host,RawGraph::Prefill);
        let before=norm::OUTPUT_FACTORIES.with(|n|n.get());
        // Privileged corruption of an actually admitted capability only; no
        // production mutator or caller-owned numeric buffer can mint this value.
        norm::FAULT.with(|n|n.set(Some(fault)));
        let result=lookup.first_bf16_norm();
        norm::FAULT.with(|n|n.set(None));
        assert!(result.is_err());
        assert_eq!(norm::OUTPUT_FACTORIES.with(|n|n.get()),before);
        assert!(host.claim_reader().is_err());
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    }
}

#[test]
fn original_lookup_norm_late_arithmetic_failure_wipes_whole_output() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let lookup=genuine_norm_input(&host,RawGraph::Prefill);
    let quote=norm::quote_for_tests(&lookup);
    let sibling=leave_primitive_work(&host,quote.work,false);
    let counter=host.inspect(|r|r.header.root.try_clone().unwrap());
    let live=counter.budget().used_bytes();
    norm::OUTPUT_DROP.with(|n|n.set(None)); norm::ROWS_DONE.with(|n|n.set(0));
    norm::OUTPUT_PREWIPE.with(|n|n.set(None));
    raw_clear_producer::NORM_GAMMA_DROP.with(|n|n.set(None));
    raw_clear_producer::LOOKUP_DROP.with(|n|n.set(None));
    norm::FAIL_ROW.with(|n|n.set(Some(2)));
    let result=lookup.first_bf16_norm();
    norm::FAIL_ROW.with(|n|n.set(None));
    assert!(result.is_err()); assert_eq!(norm::ROWS_DONE.with(|n|n.get()),2);
    assert_eq!(norm::OUTPUT_PREWIPE.with(|n|n.get()),Some((8,8,true)));
    let (wiped,covered)=norm::OUTPUT_DROP.with(|n|n.get()).unwrap();
    assert!(wiped && covered>live);
    assert!(raw_clear_producer::NORM_GAMMA_DROP.with(|n|n.get()).unwrap().0);
    assert!(raw_clear_producer::LOOKUP_DROP.with(|n|n.get()).unwrap().0);
    assert!(counter.budget().used_bytes()<live);
    host.inspect(|r| {
        sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
        assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());
    });
    assert!(host.claim_reader().is_err());
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
}

#[test]
fn cast_before_gamma_independent_core_oracle_literals() {
    // Reuse independently BigInt-certified literals from core's
    // rms_norm_bigint_every_stage_cast_before_gamma_discriminator. This is ONLY
    // a numeric component control: these slices do not claim original custody.
    // 81/128 * 65/64 = 5265/8192 rounds to 0x3f25; skipping the
    // intermediate BF16 cast gives 0x3f24. Neither comes from candidate output.
    let expected=[0x3f25,0x3fa2]; let without_intermediate_cast=0x3f24;
    let mut output=[0u16;2];
    pllm_core::field_float::rms_norm_bf16_row_into(&[0x3f80,0x4000],&[0x3f82,0x3f80],&mut output).unwrap();
    assert_eq!(output,expected); assert_ne!(output[0],without_intermediate_cast);
}

// Local positive variation of the existing synthetic fixture. Only four bytes
// change in-place, BEFORE provisioning/ingestion, with unchanged shape/size/index.
// Its ordinary source registration/lookup/norm admission funds all runtime reads,
// validation and retained gamma/output storage. This is not a post-admission
// gamma patch, a new producer constructor, or authenticated/trained source data.
fn non_power_gamma_fixture() -> Fixture {
    let mut fixture=Fixture::new();
    // Additional in-place four-byte edit and complete-file digest pass on the
    // retained ORIGINAL ledger, before either mutation or backing-byte read.
    let work=fixture.files[2].1.len().checked_add(4+128).unwrap() as u64;
    let seed=fixture.seed.as_mut().unwrap();
    seed.admit_collection(0,seed.collection_charge.payload_bytes(),work).unwrap();
    let gamma=fixture.tensors.iter().find(|t|t.name=="model.layers.0.input_layernorm.weight").unwrap();
    assert_eq!(gamma.shape,[2]); assert_eq!(gamma.dtype,"BF16");
    assert_eq!(gamma.end.checked_sub(gamma.begin),Some(4));
    let begin=fixture.data().checked_add(usize::try_from(gamma.begin).unwrap()).unwrap();
    let end=begin.checked_add(4).unwrap(); assert!(end<=fixture.files[2].1.len());
    fixture.files[2].1[begin..end].copy_from_slice(&[0x82,0x3f,0x80,0x3f]); // [65/64,1]
    // Independent synthetic premise, established before candidate ingestion.
    // Hash is conditional integrity only, never custody/source authority.
    fixture.hashes[2]=Sha256::digest(&fixture.files[2].1).into();
    fixture
}

#[test]
fn original_lookup_norm_source_gamma_cast_order_discriminator() {
    let fixture=non_power_gamma_fixture();
    let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let input=host.claim_reader().unwrap().admit_tokens(RawGraph::Prefill,&[0,0,0,0]).unwrap();
    let lookup=PrivateSourceInstance::new(input).token_lookup().unwrap();
    assert_eq!(lookup.words(),[0x3f80,0x4000,0x3f80,0x4000,0x3f80,0x4000,0x3f80,0x4000]);
    let quote=norm::quote_for_tests(&lookup);
    let sibling=leave_primitive_work(&host,quote.work,false);
    let before=(norm_counts(),norm::OUTPUT_FACTORIES.with(|n|n.get()));
    let output=lookup.first_bf16_norm().unwrap();
    assert_eq!(norm_counts(),(before.0.0+2,before.0.1+1));
    assert_eq!(norm::OUTPUT_FACTORIES.with(|n|n.get()),before.1+1);
    assert_eq!(output.shape(),[1,4,2]);
    assert!(output.original_gamma_matches_for_tests(&host,&[0x3f82,0x3f80]));
    // Reuse core's independently BigInt/rational-certified row, not a float
    // estimator or candidate-derived expectation. Ordered RN32 squares/sum/mean
    // give exactly 5 and 5/2; RN32 epsilon addition is 0x40200004. Single-round
    // rsqrt, then RN32 x*r, casts coordinate 0 to 81/128 (0x3f22).
    // (81/128)*(65/64)=5265/8192 > BF16 midpoint 5264/8192,
    // so correct coordinate is 0x3f25. Gamma before that cast (or omitted
    // intermediate cast) lies BELOW that midpoint: independent wrong result
    // 0x3f24. Coordinate 1 with gamma=1 is 0x3fa2.
    // Independent integer bounds also cover gamma-before-cast, rather than
    // assuming its RN32 reassociation equals the omitted-cast implementation.
    // t=0x40200004=2621441/1048576. Bound its exact reciprocal sqrt
    // away from endpoints by half an RN32 ulp (2^-25), BEFORE any rounding.
    let scale=1_000_000u128*(1u128<<25);
    let lo=632455u128*(1u128<<25)+1_000_000;
    let hi=632456u128*(1u128<<25)-1_000_000;
    let denominator=1_048_576u128*scale*scale;
    assert!(2_621_441u128*lo*lo<denominator);
    assert!(2_621_441u128*hi*hi>denominator);
    // Thus RN32 rsqrt lies strictly between 632455/10^6 and 632456/10^6.
    // Its BF16 cell is 0x3f22, and twice that RN32 value's cell is 0x3fa2.
    assert!(632455u64*8192>5168*1_000_000);
    assert!(632456u64*8192<5200*1_000_000);
    assert!(2*632455u64*8192>10336*1_000_000);
    assert!(2*632456u64*8192<10400*1_000_000);
    // Expand by 10^-6 (far more than RN32 product's 2^-25 error) before
    // multiplying by 65/64: the whole wrong-result interval stays STRICTLY
    // inside BF16 0x3f24's cell, between midpoints 5232/8192 and 5264/8192.
    assert!(632454u64*65*8192>5232*64*1_000_000);
    assert!(632457u64*65*8192<5264*64*1_000_000);
    let correct=[0x3f25,0x3fa2,0x3f25,0x3fa2,0x3f25,0x3fa2,0x3f25,0x3fa2];
    let wrong_gamma_before_cast=[0x3f24,0x3fa2,0x3f24,0x3fa2,0x3f24,0x3fa2,0x3f24,0x3fa2];
    assert_ne!(correct,wrong_gamma_before_cast);
    assert_eq!(output.words(),correct);
    assert_ne!(output.words(),wrong_gamma_before_cast);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
    host.inspect(|r| {
        sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
        assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());
    });
    norm::OUTPUT_DROP.with(|n|n.set(None));
    raw_clear_producer::NORM_GAMMA_DROP.with(|n|n.set(None));
    drop(output);
    assert!(norm::OUTPUT_DROP.with(|n|n.get()).unwrap().0);
    assert!(raw_clear_producer::NORM_GAMMA_DROP.with(|n|n.get()).unwrap().0);
    assert!(host.claim_reader().is_err());
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    let root=Arc::downgrade(&host.root);
    host.retire().unwrap(); host.release_final_charges().unwrap(); drop(host);
    assert!(root.upgrade().is_none());
}

#[test]
fn original_lookup_norm_genuine_foreign_reader_composed_refusal() {
    let fixture=Fixture::new();
    let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let mut lookup=genuine_norm_input(&host,RawGraph::Prefill);
    let quote=norm::quote_for_tests(&lookup);
    let other=Fixture::new().collect(19).unwrap(); other.finish().unwrap();
    let reader=other.claim_reader().unwrap();
    let sibling=leave_primitive_work(&other,quote.work,false);
    let original=leave_primitive_work(&host,1,false);
    let live=host.inspect(|r|r.header.root.budget().used_bytes());
    let foreign_live=other.inspect(|r|r.header.root.budget().used_bytes());
    // Existing privileged owner-transplant negative: actual independent sole
    // reader, not a norm capability edit, equality label or fabricated guard.
    // Production consuming entry has NO reader/plan/gamma/backing arguments.
    raw_clear_producer::replace_activation_reader_for_tests(&mut lookup,reader);
    assert_eq!(other.begin_retire().unwrap(),RetirementReadiness::Busy);
    let before=(norm_counts(),norm::OUTPUT_FACTORIES.with(|n|n.get()));
    raw_clear_producer::LOOKUP_DROP.with(|n|n.set(None));
    raw_clear_producer::TOKEN_DROP.with(|n|n.set(None));
    assert!(lookup.first_bf16_norm().is_err());
    assert_eq!((norm_counts(),norm::OUTPUT_FACTORIES.with(|n|n.get())),before);
    assert!(raw_clear_producer::LOOKUP_DROP.with(|n|n.get()).unwrap().0);
    assert!(raw_clear_producer::TOKEN_DROP.with(|n|n.get()).unwrap().0);
    assert!(host.inspect(|r|r.header.root.budget().used_bytes())<live);
    assert_eq!(other.inspect(|r|r.header.root.budget().used_bytes()),foreign_live);
    // Existing admission prepays on the actually retained reader's ledger before
    // origin/counter mismatch refuses custody. No refund/reset; original request
    // ledger is not silently substituted or debited by the foreign attempt.
    other.inspect(|r| {
        sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
        assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());
    });
    host.inspect(|r|original.prepay_primitive_work(&r.header.controller,1).unwrap());
    assert!(host.claim_reader().is_err()); assert!(other.claim_reader().is_err());
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    assert_eq!(other.begin_retire().unwrap(),RetirementReadiness::Ready);
    let roots=(Arc::downgrade(&host.root),Arc::downgrade(&other.root));
    host.retire().unwrap(); host.release_final_charges().unwrap();
    other.retire().unwrap(); other.release_final_charges().unwrap();
    drop(host); drop(other);
    assert!(roots.0.upgrade().is_none() && roots.1.upgrade().is_none());
}

#[test]
fn original_lookup_norm_equalcontent_foreign_backing_composed_refusal() {
    let fixture=Fixture::new();
    let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let lookup=genuine_norm_input(&host,RawGraph::Prefill);
    let quote=norm::quote_for_tests(&lookup);
    let other=Fixture::new().collect(19).unwrap(); other.finish().unwrap();
    let foreign_lookup=genuine_norm_input(&other,RawGraph::Prefill);
    let sibling=leave_primitive_work(&host,quote.work,false);
    let foreign=leave_primitive_work(&other,1,false);
    // Both actual headers/tensor bytes are independently ingested from the SAME
    // fixture definition, but allocations/directories/original roots differ.
    // Reuse genuinely source-funded backing; no uncharged test writer or bytes.
    let file=other.inspect(|r|Arc::clone(r.files[2].file.as_ref().unwrap()));
    let original_file=host.inspect(|r| {
        assert!(!Arc::ptr_eq(r.files[2].file.as_ref().unwrap(),&file));
        std::mem::replace(&mut r.files[2].file,Some(file))
    });
    let live=host.inspect(|r|r.header.root.budget().used_bytes());
    let before=(norm_counts(),norm::OUTPUT_FACTORIES.with(|n|n.get()));
    raw_clear_producer::LOOKUP_DROP.with(|n|n.set(None));
    raw_clear_producer::TOKEN_DROP.with(|n|n.set(None));
    assert_eq!(other.begin_retire().unwrap(),RetirementReadiness::Busy);
    assert!(lookup.first_bf16_norm().is_err());
    assert_eq!((norm_counts(),norm::OUTPUT_FACTORIES.with(|n|n.get())),before);
    assert!(raw_clear_producer::LOOKUP_DROP.with(|n|n.get()).unwrap().0);
    assert!(raw_clear_producer::TOKEN_DROP.with(|n|n.get()).unwrap().0);
    assert!(host.inspect(|r|r.header.root.budget().used_bytes())<live);
    host.inspect(|r| {
        sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
        assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());
        // Teardown ONLY, after the consuming refusal; no retry, rebind, reset or
        // successful admission uses the restored original file.
        r.files[2].file=original_file;
    });
    other.inspect(|r|foreign.prepay_primitive_work(&r.header.controller,1).unwrap());
    assert!(host.claim_reader().is_err()); assert!(other.claim_reader().is_err());
    assert_eq!(other.begin_retire().unwrap(),RetirementReadiness::Busy);
    raw_clear_producer::LOOKUP_DROP.with(|n|n.set(None));
    raw_clear_producer::TOKEN_DROP.with(|n|n.set(None));
    drop(foreign_lookup);
    assert!(raw_clear_producer::LOOKUP_DROP.with(|n|n.get()).unwrap().0);
    assert!(raw_clear_producer::TOKEN_DROP.with(|n|n.get()).unwrap().0);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    assert_eq!(other.begin_retire().unwrap(),RetirementReadiness::Ready);
    let roots=(Arc::downgrade(&host.root),Arc::downgrade(&other.root));
    host.retire().unwrap(); host.release_final_charges().unwrap();
    other.retire().unwrap(); other.release_final_charges().unwrap();
    drop(host); drop(other);
    assert!(roots.0.upgrade().is_none() && roots.1.upgrade().is_none());
}
