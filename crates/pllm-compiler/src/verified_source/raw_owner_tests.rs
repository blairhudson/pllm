//! Synthetic in-memory definitions ONLY. Authored, not execution evidence.
#[path = "../replicated_decoder/raw_first_norm_tests.rs"]
mod first_bf16_norm;
use super::*;
use super::raw_fixture::{experiment, Fixture, Tensor, CONFIG};
impl Fixture {
    fn started(self) -> TestHost {
        let host=TestHost(RawHost::prepare(self.header()).unwrap());
        host.start(snapshot_directory::RawSnapshotDirectory::parent_device().unwrap()).unwrap();
        host
    }
    fn collect(self,chunk:usize) -> Result<TestHost,Error> {
        let registration=self.started();
        registration.ingest_fixture(chunk)?;
        Ok(registration)
    }
    // Only format-negative fixtures repin deliberately malformed SYNTHETIC
    // bytes as host premises, so the format check, not the whole hash, is tested.
    fn repin(&mut self,file:usize) {
        self.sizes[file]=self.files[file].1.len() as u64;
        self.hashes[file]=Sha256::digest(&self.files[file].1).into();
    }
    fn rewrite_header(&mut self,edit:impl FnOnce(&mut serde_json::Value)) {
        let old=u64::from_le_bytes(self.files[2].1[..8].try_into().unwrap()) as usize;
        let mut json:serde_json::Value=serde_json::from_slice(&self.files[2].1[8..8+old]).unwrap();
        edit(&mut json);
        let json=serde_json::to_vec(&json).unwrap();
        let mut bytes=(json.len() as u64).to_le_bytes().to_vec(); bytes.extend(json);
        bytes.extend_from_slice(&self.files[2].1[8+old..]); self.files[2].1=bytes; self.repin(2);
    }
    fn data(&self) -> usize { 8+u64::from_le_bytes(self.files[2].1[..8].try_into().unwrap()) as usize }

    fn d32() -> Self {
        let mut config:serde_json::Value=serde_json::from_slice(CONFIG).unwrap();
        config["quantization_config"]=serde_json::json!({"quant_method":"fp8","fmt":"e4m3","weight_block_size":[128,128]});
        let mut fixture=Self::from_config(&serde_json::to_vec(&config).unwrap());
        let body_cap=fixture.tensors.last().unwrap().begin as usize+16384+2;
        fixture.reserve_edit(fixture.tensors.len()+1,body_cap).unwrap();
        fixture.recipe="D32".into();
        let data=fixture.data();
        let unused=fixture.tensors.last_mut().unwrap();
        let mut body=exact_vec(body_cap).unwrap();
        body.extend_from_slice(&fixture.files[2].1[data..data+unused.begin as usize]);
        unused.shape=vec![128,128]; unused.dtype="F8_E4M3".into();
        unused.end=unused.begin+16384;
        unused.scale=Some("zz_unused.weight_scale_inv".into());
        body.extend(std::iter::repeat_n(0x38,16384)); // independent synthetic FP8 ones
        let begin=unused.end;
        fixture.tensors.push(Tensor {name:"zz_unused.weight_scale_inv".into(),shape:vec![1,1],
            dtype:"BF16".into(),begin,end:begin+2,scale:None});
        body.extend_from_slice(&0x3f80u16.to_le_bytes());
        fixture.reemit(body).unwrap();
        fixture
    }
}
fn counts() -> (usize,usize) { (HEADER_FACTORIES.with(|c|c.get()),WRITER_FACTORIES.with(|c|c.get())) }

// RawSnapshotDirectory::parent_device/check_file currently require Unix.
// Core hook itself is portable; no Windows native-EOF coverage claimed here.
#[cfg(unix)]
#[test]
fn genuine_snapshot_payload_eof_retains_old_window() {
    let fixture=Fixture::new(); // independently owned SYNTHETIC source only
    let counter=fixture.seed.as_ref().unwrap().root.try_clone().unwrap();
    let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let key=repair_key(&host);
    let reader=host.claim_reader().unwrap();
    let tensor=reader.weight(RawGraph::Prefill,key.operation).unwrap();
    let without_window=counter.budget().used_bytes();
    let older=tensor.read(2,2).unwrap();
    assert_eq!(older.bytes(),[0,0x40]); // independent synthetic BF16 2.0
    let window_charge=independent_window_bytes()+size_of::<OwnedExternalCharge>();
    let live=counter.budget().used_bytes();
    assert_eq!(live,without_window+window_charge);
    // Leave EXACT independently priced read bootstrap + full demand work.
    // This sibling shares the original ledger; no cap/sentinel is changed.
    let sibling=leave_reader_work(&host,1+8*4096+1024+256+4,false);
    let live=counter.budget().used_bytes(); // sibling's own controls included
    let cover=host.inspect(|r|r.header.collection_charge.payload_bytes());
    let offset=0u64;
    let len=2usize;
    {
        // Resolve ONLY the selected genuine snapshot under its actual host lock.
        let state=host.root.state.lock().unwrap();
        let r=state.payload.as_ref().unwrap();
        let expected=&r.header.raw_premise.as_ref().unwrap().tensors[tensor.tensor];
        let backing=&r.files[expected.file];
        let file=backing.file.as_ref().unwrap();
        assert!(Arc::ptr_eq(&backing.original.upgrade().unwrap(),file));
        let start=data_start(file).unwrap();
        let b=start.checked_add(expected.begin).unwrap().checked_add(offset).unwrap();
        assert!(len>=2 && len<=4096);
        assert!(offset+len as u64<=expected.end-expected.begin);
        assert!(b>=start && b+len as u64<=file.size());
        let sealed_size=file.size(); let sealed_digest=file.digest();
        // Non-shorter lengths refuse before any set_len; cached geometry stays.
        for rejected in [sealed_size,sealed_size+1] {
            assert_eq!(file.truncate_snapshot_for_tests(rejected).unwrap_err().kind(),
                std::io::ErrorKind::InvalidInput);
        }
        file.truncate_snapshot_for_tests(b+1).unwrap();
        assert!(b+2<sealed_size);
        for rejected in [b+1,b+2] {
            assert_eq!(file.truncate_snapshot_for_tests(rejected).unwrap_err().kind(),
                std::io::ErrorKind::InvalidInput);
        }
        assert_eq!(file.size(),sealed_size); assert_eq!(file.digest(),sealed_digest);
        assert_eq!(data_start(file).unwrap(),start); // header still intact
    } // release host mutex BEFORE tensor.read: no recursive lock acquisition
    assert_eq!(READER_IO_FAULT.with(|n|n.get()),None);
    reset_repair_probes(); WINDOW_DROP.with(|n|n.set(None));
    let factories=WINDOW_FACTORIES.with(|n|n.get());
    // Cached bounds pass. Native positional read gets one byte (0x80), then EOF.
    // Failure itself uses neither injected read fault nor synthetic error path.
    let failed=tensor.read(offset,len);
    assert!(matches!(failed,Err(Error::Binding)));
    assert_eq!(WINDOW_FACTORIES.with(|n|n.get()),factories+1);
    assert_eq!(repair_probes()[3],1); assert_eq!(repair_probes()[4],1);
    assert_eq!(WINDOW_DROP.with(|n|n.get()),Some((true,live+window_charge)));
    // Existing observer checks ALL FRAME bytes while failed window still charged.
    assert_eq!(counter.budget().used_bytes(),live);
    assert_eq!(older.bytes(),[0,0x40]); // old data/charge survive failed read
    drop(failed);
    assert_eq!(counter.budget().used_bytes(),live);
    assert!(host.root.state.lock().unwrap().reader_failed);
    reset_repair_probes();
    assert!(matches!(tensor.read(0,2),Err(Error::Retired)));
    assert!(reader.weight(RawGraph::Prefill,key.operation).is_err());
    assert_eq!(repair_probes(),[0;7]);
    assert_eq!(counter.budget().used_bytes(),live);
    assert_eq!(older.bytes(),[0,0x40]);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
    host.inspect(|r| {
        assert_eq!(r.header.collection_charge.payload_bytes(),cover);
        sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
    });
    drop(older);
    assert_eq!(counter.budget().used_bytes(),live-window_charge);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
    drop(tensor); drop(reader);
    let retained=counter.budget().used_bytes();
    assert_eq!(retained,live-window_charge-independent_tensor_bytes()-size_of::<OwnedExternalCharge>());
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    host.inspect(|r|assert_eq!(r.header.collection_charge.payload_bytes(),cover));
    host.retire().unwrap();
    assert_eq!(counter.budget().used_bytes(),retained);
    host.inspect(|r| {
        assert_eq!(r.header.collection_charge.payload_bytes(),cover);
        // Full failed-read work stayed spent through owner Drop and retirement.
        sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
        assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());
    });
    host.release_final_charges().unwrap();
    assert!(counter.budget().used_bytes()<retained);
    drop(host); drop(sibling);
    assert_eq!(counter.budget().used_bytes(),OwnedBudget::resources().unwrap().initial_reserved_bytes);
}

#[test]
fn consuming_fixture_ingestion_keeps_actual_factory_source_counter_and_ledger() {
    let fixture=Fixture::new();
    let seed=fixture.seed.as_ref().unwrap(); let plan=seed.original.plan() as *const _;
    let counter=seed.root.try_clone().unwrap();
    let bootstrap=seed.original.lower_bootstrap_bytes(seed.limits).unwrap();
    let sibling=seed.original.lower(seed.root.charge_external(bootstrap).unwrap(),seed.original.workload(),seed.limits).unwrap();
    let before=sibling.resources().prepaid_work;
    let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    host.inspect(|r| {
        assert!(std::ptr::eq(counter.budget(),r.header.original.budget()));
        assert!(std::ptr::eq(plan,r.header.original.plan())); assert!(sibling.same_origin(&r.header.original));
        let capture=r.header.fixture.as_ref().unwrap(); assert!(capture.seed.is_none());
        assert_eq!(capture.files.len(),3); assert_eq!(r.files.len(),3);
        let bootstrap=r.header.original.lower_bootstrap_bytes(r.header.limits).unwrap();
        let after=r.header.original.lower(r.header.root.charge_external(bootstrap).unwrap(),r.header.original.workload(),r.header.limits).unwrap();
        assert!(after.resources().prepaid_work>before);
    });
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    host.retire().unwrap(); host.release_final_charges().unwrap(); drop(host); drop(sibling);
    assert_eq!(counter.budget().used_bytes(),OwnedBudget::resources().unwrap().initial_reserved_bytes);
}

// Test-only explicit cleanup, never production Drop-based retirement. Failed
// retire deliberately leaves the native retention roots intact.
struct TestHost(RawHost);
impl std::ops::Deref for TestHost {
    type Target=RawHost;
    fn deref(&self) -> &RawHost { &self.0 }
}
impl RawHost {
    // Narrow ingestion helpers only. The original append gate holds its mutex
    // across every payload borrow; concurrent retirement cannot release cover.
    fn ingest_fixture(&self,chunk:usize) -> Result<(),Error> {
        let files=self.inspect(|r|r.header.fixture.as_ref().unwrap().files.len());
        for file in 0..files {
            let len=self.inspect(|r|r.header.fixture.as_ref().unwrap().files[file].1.len());
            for offset in (0..len).step_by(chunk) {
                self.append_fixture(file,offset as u64,offset..offset.saturating_add(chunk).min(len))?;
            }
        }
        Ok(())
    }
    fn append_fixture(&self,file:usize,offset:u64,range:std::ops::Range<usize>) -> Result<(),Error> {
        self.append_with(|r| {
            let fixture=r.header.fixture.take().unwrap();
            let result=r.append(file,offset,&fixture.files[file].1[range]);
            r.header.fixture=Some(fixture); result
        })
    }
    fn append_fixture_file(&self,file:usize) -> Result<(),Error> {
        let len=self.inspect(|r|r.header.fixture.as_ref().unwrap().files[file].1.len());
        self.append_fixture(file,0,0..len)
    }
}
impl Drop for TestHost {
    fn drop(&mut self) {
        if self.0.retire().is_ok() { self.0.release_final_charges().unwrap(); }
    }
}
impl RawHost {
    fn inspect<T>(&self, f: impl FnOnce(&mut RawRegistration)->T) -> T {
        let mut state=self.root.state.lock().unwrap();
        f(state.payload.as_mut().unwrap())
    }
}

#[test]
fn missing_authority_before_factories() {
    let before=counts();
    assert!(SourceHeaderEntry::begin_raw_host(b"not parsed",b"not parsed",4,2,[1;32],[2;32],[3;32],None).is_err());
    assert_eq!(counts(),before);
    let legacy=SourceHeaderEntry::begin_mode(&experiment(),CONFIG,4,2,[1;32],[2;32],[3;32],true).unwrap();
    // A test can forge the private capsule, but a legacy header has no premise.
    let directory=snapshot_directory::RawSnapshotDirectory::prepare(&legacy.original,&legacy.controller,&legacy.root).unwrap();
    let header=Arc::new(legacy); let identity=Arc::downgrade(&header);
    let before=counts(); assert!(RawHost::prepare(RawPending {fixture:None,header,identity,directory}).is_err()); assert_eq!(counts(),before);
}
#[test]
fn over_4gib_before_factories() {
    assert_eq!(public_file::MAX_BYTES,4_294_967_296);
    let files=[("config.json",1,[1;32]),("model.safetensors.index.json",1,[2;32]),("selected.safetensors",4_412_140_424,[3;32])];
    let tensors=[("x",2,"BF16",&[1][..],0,2,None)];
    let e=HostRawExpectations::provision(&files,&tensors,[1;32],[4;32],"D31");
    let before=counts();
    assert!(SourceHeaderEntry::begin_raw_host(b"not parsed",b"not parsed",4,2,[1;32],[2;32],[3;32],Some(e)).is_err());
    assert_eq!(counts(),before);
}
#[test]
fn oversized_descriptor_count_before_factories() {
    let files=vec![("not traversed",1,[1;32]);source_policy::DESCRIPTORS+1];
    let tensors=[("x",2,"BF16",&[1][..],0,2,None)];
    let before=counts();
    let e=HostRawExpectations::provision(&files,&tensors,[1;32],[2;32],"D31");
    assert!(SourceHeaderEntry::begin_raw_host(b"bad",b"bad",4,2,[1;32],[2;32],[3;32],Some(e)).is_err());
    assert_eq!(counts(),before);
}
#[test]
fn mixed_original_owners_rejected() {
    let mut a=Fixture::new().header(); let mut b=Fixture::new().header();
    // Deliberately bypass the inaccessible immutable capsule in this child test.
    a.identity=std::sync::Weak::new(); b.identity=std::sync::Weak::new();
    std::mem::swap(&mut Arc::get_mut(&mut a.header).unwrap().original,&mut Arc::get_mut(&mut b.header).unwrap().original);
    a.identity=Arc::downgrade(&a.header);
    let before=counts(); assert!(RawHost::prepare(a).is_err()); assert_eq!(counts(),before);
}
#[test]
fn substituted_header_rejected() {
    let registration=Fixture::new().collect(17).unwrap();
    // Whole replacement header is individually valid, same config/plan/pins.
    // Registration's ORIGINAL private identity must still reject this join.
    registration.inspect(|r| r.header=Fixture::new().header());
    assert!(registration.finish().is_err());
    let registration=Fixture::new().started();
    registration.inspect(|r| r.header=Fixture::new().header());
    assert!(registration.append_fixture_file(0).is_err());
}
#[test]
fn payload_cannot_replace_premise() {
    let mut fixture=Fixture::new();
    let mut index:serde_json::Value=serde_json::from_slice(&fixture.files[1].1).unwrap();
    index["expected_files"]=serde_json::json!({"synthetic-00001.safetensors":"attacker pin"});
    fixture.files[1].1=serde_json::to_vec(&index).unwrap(); fixture.repin(1);
    assert!(fixture.collect(31).unwrap().finish().is_err());
}
#[test]
fn unselected_corruption_rejected() {
    let mut fixture=Fixture::new();
    assert_eq!(fixture.tensors.last().unwrap().name,"zz_unused.weight");
    *fixture.files[2].1.last_mut().unwrap()^=1;
    assert!(fixture.collect(7).is_err()); // complete-file hash, before finish/ranges
}
#[test]
fn selected_corruption_rejected() {
    let mut fixture=Fixture::new(); let offset=fixture.data()+fixture.tensors[0].begin as usize;
    fixture.files[2].1[offset]^=1;
    assert!(fixture.collect(19).is_err());
}
#[test]
fn substituted_backing_rejected() {
    let fixture=Fixture::new(); let registration=fixture.collect(13).unwrap();
    // Independently completed SAME bytes/hash/length, even same directory. No
    // digest equality or fstat observation may replace private original backing.
    registration.inspect(|r| {
        let fixture=r.header.fixture.as_ref().unwrap();
        let mut writer=PublicFileWriter::new_in(fixture.sizes[0],fixture.hashes[0],Arc::clone(&r.directory) as Arc<dyn SnapshotDirectory>).unwrap();
        writer.write(&fixture.files[0].1).unwrap();
        r.files[0].file=Some(Arc::new(writer.finish().unwrap()));
    });
    assert!(registration.finish().is_err());
}
#[test]
fn failed_append_retains_registration_and_refuses_retry() {
    let fixture=Fixture::new(); let registration=fixture.started();
    let (directory,counter)=registration.inspect(|r|(Arc::downgrade(&r.directory),r.header.root.try_clone().unwrap()));
    let live=counter.budget().used_bytes();
    let _:fn(&RawHost,usize,u64,&[u8])->Result<(),Error>=RawHost::append;
    let _:fn(&RawHost)->Result<(),Error>=RawHost::finish;
    assert!(registration.append(0,1,b"x").is_err());
    assert!(registration.append_fixture_file(0).is_err());
    assert!(registration.finish().is_err());
    assert!(directory.upgrade().is_some()); assert_eq!(counter.budget().used_bytes(),live);
    registration.retire().unwrap();
    assert_eq!(counter.budget().used_bytes(),live); // future outer lease still live
    registration.release_final_charges().unwrap();
    assert!(directory.upgrade().is_none()); assert!(counter.budget().used_bytes()<live);
}
#[test]
fn backing_drops_before_charge() {
    let fixture=Fixture::new(); let registration=fixture.collect(23).unwrap();
    let (counter,cover)=registration.inspect(|r| {
        let counter=r.header.root.try_clone().unwrap();
        let cover=r.header.collection_charge.charged_bytes();
        for file in &mut r.files { file.observe=Some((counter.try_clone().unwrap(),cover)); }
        (counter,cover)
    });
    BACKING_DROP.with(|s|s.set(None));
    assert!(registration.append(3,0,b"trailing").is_err());
    assert!(BACKING_DROP.with(|s|s.get()).is_none());
    registration.retire().unwrap();
    let (closed,used,expected)=BACKING_DROP.with(|s|s.take()).unwrap();
    assert!(closed); assert_eq!(expected,cover); assert!(used>=cover);
    assert!(counter.budget().used_bytes()>=cover);
    registration.release_final_charges().unwrap();
    assert!(counter.budget().used_bytes()<cover);
}
#[test]
fn exact_one_short_phase_bytes_before_writer() {
    for short in [true,false] {
        let mut owned=Fixture::new().into_collector().unwrap();
        let SourceCollector {payload,header}=&mut owned;
        let fixture=payload.as_ref().unwrap();
        let generation=header.collection_charge.payload_bytes();
        let phase=independent_phase_bytes(header,fixture.files.len(),fixture.tensors.len());
        let mut outside=header.root.charge_external(0).unwrap();
        // The independent premise identity is live alongside the full phase
        // charge. Its try_clone reserves an additional inline OwnedBudget slot.
        let exact=MAX_PROGRAM_BYTES-header.root.budget().used_bytes()-(phase-generation)-size_of::<OwnedBudget>();
        outside.resize_payload(exact+usize::from(short)).unwrap();
        let before=counts();
        let result=fixture.with_expected(|e|TrustedRawPremise::provision(header,e)).and_then(|result|result);
        assert_eq!(result.is_err(),short);
        assert_eq!(counts(),before);
        if short {
            assert!(header.failed); assert!(header.collection_charge.payload_bytes()<phase);
        } else {
            assert_eq!(header.collection_charge.payload_bytes(),phase);
            assert_eq!(header.root.budget().used_bytes(),MAX_PROGRAM_BYTES);
        }
        drop(result); drop(outside);
    }
}
#[test]
fn preparation_charges_both_late_shared_scale_searches_on_original_ledger() {
    fn enlarged() -> Fixture {
    let mut fixture=Fixture::d32();
    fixture.tensors.truncate(fixture.tensors.len()-2);
    let base=fixture.tensors.len();
    let count=256usize;
    fixture.reserve_edit(base+count+1,0).unwrap(); // metadata-only; no larger body
    let prefix="x".repeat(NAME-4);
    let scale=format!("{prefix}ffff");
    let mut end=fixture.tensors.last().unwrap().end;
    for i in 0..count {
        let begin=end; end+=16384;
        fixture.tensors.push(Tensor {name:format!("{prefix}{i:04x}"),shape:vec![128,128],
            dtype:"F8_E4M3".into(),begin,end,scale:Some(scale.clone())});
    }
    let begin=end; end+=2;
    fixture.tensors.push(Tensor {name:scale,shape:vec![1,1],dtype:"BF16".into(),begin,end,scale:None});
    // Metadata-only host expectations. Never construct/write the larger shard.
    fixture.sizes[2]=fixture.data() as u64+end;
    fixture
    }
    let fixture=enlarged();
    let count=256usize; let base=fixture.tensors.len()-count-1;
    let t=fixture.tensors.len();
    let f=fixture.files.len();
    let seed=fixture.seed.as_ref().unwrap();
    let ops=seed.original.plan().prefill.operations.len();
    // Independent observed-loop lower bound: new-name dedup alone visits the
    // earlier long names; BOTH scale searches reach the final shared scale.
    // Count only their guaranteed common-prefix bytes, ignoring all other work.
    let long_dedup=count*(count+1)/2;
    let scale_searches=2*count*(count+1);
    assert!((long_dedup+scale_searches)*(NAME-4)>(f+t+ops)*(f+t+ops)*NAME);
    // Independent closed-form preparation debit, including the unchanged
    // bootstrap/copy allowance and the later configuration parser's full cap.
    // These are declared logical debits, NOT measured comparison/CPU work.
    let metadata=f*(size_of::<ExpectedFile>()+NAME)+t*(size_of::<ExpectedTensor>()+2*NAME);
    let shapes=ops*(source_registry::tree_entry_bytes::<(String,Vec<usize>)>().unwrap()
        +source_registry::tree_entry_bytes::<(String,usize)>().unwrap()+2*NAME+2*size_of::<usize>());
    let controls=3*size_of::<TrustedRawPremise>()+size_of::<OwnedBudget>()+3*size_of::<HostRawExpectations<'_>>();
    let initial=seed.collection_charge.payload_bytes()+independent_registration_reader_io_bootstrap_bytes()+2*source_policy::CONFIG_BYTES+2*metadata+2*shapes+controls
        +canonical::resources::<std::io::Sink>(16).unwrap().hash_workspace_bytes;
    let baseline=2*CANONICAL_WORK+(f+t+ops)*(f+t+ops)*NAME+initial
        +4*source_policy::CONFIG_BYTES+1_000_000_000;
    let scale_passes=2*t*t*NAME;
    let required=u64::try_from(baseline+scale_passes).unwrap();
    assert!(baseline>=2*CANONICAL_WORK+1_000_000_000);
    let directories=snapshot_directory::OwnedSnapshotDirectory::factories_for_tests();
    for short in [true,false] {
        let mut owned=enlarged().into_collector().unwrap();
        let SourceCollector {payload,header}=&mut owned;
        let fixture=payload.as_ref().unwrap();
        let factories=counts();
        let bootstrap=header.original.lower_bootstrap_bytes(header.limits).unwrap();
        let sibling=header.original.lower(header.root.charge_external(bootstrap).unwrap(),header.original.workload(),header.limits).unwrap();
        assert!(sibling.same_origin(&header.original));
        let remaining=required-u64::from(short);
        let fill=header.limits.max_work-sibling.resources().prepaid_work-remaining;
        header.original.prepay_compiler_metadata(&header.controller,fill).unwrap();
        let result=fixture.with_expected(|e|TrustedRawPremise::provision(header,e)).and_then(|result|result);
        if short {
            // OLD pays only baseline: remaining-baseline=scale_passes-1>0,
            // so it admits this case and fails this assertion. NEW reaches the
            // configuration prepay with its entire 1e9 cap funded except ONE.
            assert!(scale_passes>1); assert!(result.is_err());
            assert!(header.failed);
            assert_eq!(header.collection_charge.payload_bytes(),initial);
            assert!(sibling.prepay_compiler_metadata(&header.controller,0).is_err());
        } else {
            let premise=result.unwrap();
            assert_eq!(premise.tensors.len(),base+count+1);
            for tensor in &premise.tensors[base..base+count] {
                assert_eq!(tensor.scale_index,Some(t-1));
            }
            assert!(std::ptr::eq(premise.identity.budget(),header.original.budget()));
            // Exact cumulative debit exhausts the SAME original quota. OLD
            // leaves scale_passes unused and wrongly accepts the extra unit.
            sibling.prepay_compiler_metadata(&header.controller,0).unwrap();
            assert!(sibling.prepay_compiler_metadata(&header.controller,1).is_err());
        }
        assert_eq!(counts(),factories);
    }
    // Quota boundary is logical preparation work, not an ingestion/FS factory.
    assert_eq!(snapshot_directory::OwnedSnapshotDirectory::factories_for_tests(),directories);
}

#[test]
fn exact_one_short_original_work_and_failed_attempt_burns_siblings() {
    for short in [true,false] {
        let fixture=Fixture::new(); let header=fixture.header();
        // Actual original-source sibling, not a copied plan or fresh ledger.
        // Its resource summary records the shared cumulative used total at lower.
        let bootstrap=header.original.lower_bootstrap_bytes(header.limits).unwrap();
        let sibling=header.original.lower(header.root.charge_external(bootstrap).unwrap(),header.original.workload(),header.limits).unwrap();
        assert!(sibling.same_origin(&header.original));
        let control=header.root.charge_external(OriginalSourcePlan::compiler_metadata_control_bytes().unwrap()).unwrap();
        let required=header.raw_premise.as_ref().unwrap().work;
        let fill=header.limits.max_work-sibling.resources().prepaid_work-required+u64::from(short);
        header.original.prepay_compiler_metadata(&control,fill).unwrap();
        let before=WRITER_FACTORIES.with(|n|n.get());
        let result=RawHost::prepare(header);
        assert_eq!(result.is_err(),short);
        assert_eq!(WRITER_FACTORIES.with(|n|n.get()),before); // preparation stays FS-free
        for _ in 0..3 { assert_eq!(sibling.prepay_compiler_metadata(&control,0).is_err(),short); }
        drop(result);
    }
}
#[test]
fn exact_chunk_inventory_boundaries_replay_and_exhaustion() {
    for chunk in [1,2,7,4096,public_file::MAX_CHUNK_BYTES] {
        let owner=Fixture::new().collect(chunk).unwrap(); owner.finish().unwrap();
        owner.inspect(|r| {
            assert_eq!(r.files.len(),3); check_header(&r.header,&r.directory,&r.identity).unwrap();
            assert!(r.header.raw_premise.is_some());
            assert_eq!(r.files[2].digest(),r.header.fixture.as_ref().unwrap().hashes[2]);
        });
        assert!(owner.finish().is_err()); // no repeated inventory pass
    }
    for (file,offset,bytes) in [(1,0,b"x".as_slice()),(0,0,b"".as_slice()),(0,u64::MAX,b"x".as_slice())] {
        assert!(Fixture::new().started().append(file,offset,bytes).is_err());
    }
    let registration=Fixture::new().started(); registration.append_fixture(0,0,0..1).unwrap();
    assert!(registration.append_fixture(0,0,0..1).is_err());
    assert!(Fixture::new().started().finish().is_err());
    let too_large=vec![0;public_file::MAX_CHUNK_BYTES+1];
    assert!(Fixture::new().started().append(0,0,&too_large).is_err());
}
#[test]
fn inventory_duplicates_overlap_reordering_missing_and_binding_refuse() {
    for case in 0..6 {
        let mut fixture=Fixture::new();
        match case {
            0=>fixture.tensors[1].name=fixture.tensors[0].name.clone(),
            1=>fixture.tensors[1].begin-=1,
            2=>fixture.tensors.swap(0,1),
            3=>{ fixture.tensors.pop(); },
            4=>fixture.semantic[0]^=1,
            _=>fixture.hashes[0][0]^=1,
        }
        if case==3 { assert!(fixture.collect(11).unwrap().finish().is_err()); }
        else { let before=WRITER_FACTORIES.with(|n|n.get()); assert!(fixture.header_result().is_err()); assert_eq!(WRITER_FACTORIES.with(|n|n.get()),before); }
    }
}
#[test]
fn verified_header_dtype_shape_offsets_duplicates_and_trailing_refuse() {
    for case in 0..6 {
        let mut fixture=Fixture::new(); let name=fixture.tensors[0].name.clone();
        match case {
            0=>fixture.rewrite_header(|v|v[&name]["dtype"]=serde_json::json!("F16")),
            1=>fixture.rewrite_header(|v|v[&name]["shape"]=serde_json::json!([999,2])),
            2=>fixture.rewrite_header(|v|v[&name]["data_offsets"]=serde_json::json!([u64::MAX,0])),
            3=>{fixture.files[2].1.push(0);fixture.repin(2);},
            4=>{
                let old=fixture.data(); let header=std::str::from_utf8(&fixture.files[2].1[8..old]).unwrap();
                let mut h=header[..header.len()-1].to_owned();
                h.push(',');h.push_str(&serde_json::to_string(&name).unwrap());h.push(':');
                let parsed:serde_json::Value=serde_json::from_str(header).unwrap(); h.push_str(&serde_json::to_string(&parsed[&name]).unwrap());h.push('}');
                let mut bytes=(h.len() as u64).to_le_bytes().to_vec();bytes.extend(h.as_bytes());bytes.extend_from_slice(&fixture.files[2].1[old..]);fixture.files[2].1=bytes;fixture.repin(2);
            },
            _=>{fixture.files[2].1[..8].copy_from_slice(&u64::MAX.to_le_bytes());fixture.repin(2);},
        }
        assert!(fixture.collect(29).unwrap().finish().is_err());
    }
}
#[test]
fn finite_head_tie_and_index_same_verified_bytes() {
    for case in 0..4 {
        let mut fixture=Fixture::new();
        if case==3 {
            let mut index:serde_json::Value=serde_json::from_slice(&fixture.files[1].1).unwrap();
            index["weight_map"][&fixture.tensors[0].name]=serde_json::json!("foreign.safetensors");
            fixture.files[1].1=serde_json::to_vec(&index).unwrap(); fixture.repin(1);
        } else {
            let name=if case==2 { semantic_weight(fixture.seed.as_ref().unwrap(),pllm_models::ModelOperator::OutputHead).unwrap().to_owned() } else {"zz_unused.weight".into()};
            let tensor=fixture.tensors.iter().find(|t|t.name==name).unwrap();
            let offset=fixture.data()+tensor.begin as usize;
            let bits=match case {0=>0x7f80u16,1=>0x7fc1,_=>0x4000};
            fixture.files[2].1[offset..offset+2].copy_from_slice(&bits.to_le_bytes()); fixture.repin(2);
        }
        assert!(fixture.collect(37).unwrap().finish().is_err());
    }
}
#[test]
fn scale_relationships_before_writer_no_bf16_to_fp8_conversion() {
    for case in 0..4 {
        let mut fixture=Fixture::new();
        let t=fixture.tensors.last_mut().unwrap();
        match case {
            0=>t.scale=Some("missing.weight_scale_inv".into()),
            1=>t.dtype="F8_E4M3".into(), // D31 may not silently adopt D32
            2=>{ t.dtype="F16".into(); },
            _=>{fixture.recipe="D32".into();}, // missing exact FP8 config
        }
        let before=WRITER_FACTORIES.with(|n|n.get());
        assert!(fixture.header_result().is_err()); assert_eq!(WRITER_FACTORIES.with(|n|n.get()),before);
    }
}
#[test]
fn raw_header_cannot_enter_legacy_collection_or_q18_import() {
    fn unpack_for_negative_test(pending:RawPending) -> SourceHeaderEntry {
        let RawPending {fixture,header,identity,directory}=pending;
        drop(fixture); drop(identity); drop(directory);
        Arc::try_unwrap(header).ok().unwrap()
    }
    assert!(unpack_for_negative_test(Fixture::new().header()).collect(0,0,0,|_,_| -> Result<(),Error> {panic!("raw header entered legacy factory")}).is_err());
    assert!(unpack_for_negative_test(Fixture::new().header()).into_compact_import().is_err());
}
#[test]
fn duplicate_counter_includes_escaped_and_nested_keys() {
    // The same member-count invariant is applied only AFTER the existing C9
    // parser validated syntax. These constants independently exercise its logic.
    for bytes in [br#"{"x":1,"x":2}"#.as_slice(),br#"{"x":[{"y":0,"\u0079":1}]}"#,br#"{"x":{"a":1},"x":0}"#] {
        let parsed:serde_json::Value=serde_json::from_slice(bytes).unwrap();
        assert!(source_members(bytes)>retained_members(&parsed));
    }
    let bytes=br#"{"x":"escaped\\\" colon: not a member","y":[{"z":3}]}"#;
    let parsed:serde_json::Value=serde_json::from_slice(bytes).unwrap();
    assert_eq!(source_members(bytes),retained_members(&parsed));
}

#[test]
fn pending_and_host_preparation_are_filesystem_free_and_fully_charged() {
    let fixture=Fixture::new();
    let directories=snapshot_directory::OwnedSnapshotDirectory::factories_for_tests();
    let writers=WRITER_FACTORIES.with(|n|n.get());
    let pending=fixture.header();
    assert!(pending.header.directory.is_none());
    let phase=pending.raw_premise.as_ref().unwrap().phase_bytes;
    assert_eq!(pending.collection_charge.payload_bytes(),phase);
    assert!(Arc::ptr_eq(&pending.header,&pending.identity.upgrade().unwrap()));
    let host=TestHost(RawHost::prepare(pending).unwrap());
    host.inspect(|r| {
        assert_eq!(r.header.collection_charge.payload_bytes(),phase);
        assert!(r.writer.is_none()); assert!(r.files.is_empty());
        assert_eq!(r.directory.retention_for_tests(),(false,false,false));
    });
    assert_eq!(snapshot_directory::OwnedSnapshotDirectory::factories_for_tests(),directories);
    assert_eq!(WRITER_FACTORIES.with(|n|n.get()),writers);
}

#[test]
fn pending_retire_and_external_drop_retain_until_explicit_release() {
    let fixture=Fixture::new();
    let directories=snapshot_directory::OwnedSnapshotDirectory::factories_for_tests();
    let writers=WRITER_FACTORIES.with(|n|n.get());
    let host=RawHost::prepare(fixture.header()).unwrap();
    let (header,directory,counter)=host.inspect(|r|(
        r.header.identity.clone(),Arc::downgrade(&r.directory),r.header.root.try_clone().unwrap(),
    ));
    let live=counter.budget().used_bytes();
    let root=Arc::downgrade(&host.root);
    assert!(host.release_final_charges().is_err());
    host.retire().unwrap(); host.retire().unwrap(); // successful retirement is idempotent
    drop(host);
    assert!(root.upgrade().is_some()); assert!(header.upgrade().is_some());
    assert!(directory.upgrade().is_some()); assert_eq!(counter.budget().used_bytes(),live);
    // Recover only after dropping the sole external handle: no strong test root
    // may mask missing native retention. No real lease exists in this test.
    let host=RawHost {root:root.upgrade().unwrap()};
    host.retire().unwrap();
    assert_eq!(counter.budget().used_bytes(),live);
    host.release_final_charges().unwrap();
    assert!(header.upgrade().is_none()); assert!(directory.upgrade().is_none());
    assert_eq!(counter.budget().used_bytes(),OwnedBudget::resources().unwrap().initial_reserved_bytes);
    assert!(host.release_final_charges().is_err()); assert!(host.retire().is_err());
    drop(host); assert!(root.upgrade().is_none());
    assert_eq!(snapshot_directory::OwnedSnapshotDirectory::factories_for_tests(),directories);
    assert_eq!(WRITER_FACTORIES.with(|n|n.get()),writers);
}

#[test]
fn ingestion_before_start_and_external_drop_retain_until_explicit_release() {
    let directories=snapshot_directory::OwnedSnapshotDirectory::factories_for_tests();
    let writers=WRITER_FACTORIES.with(|n|n.get());
    for finish in [false,true] {
        let host=RawHost::prepare(Fixture::new().header()).unwrap();
        let (header,directory,counter)=host.inspect(|r|(
            r.header.identity.clone(),Arc::downgrade(&r.directory),r.header.root.try_clone().unwrap(),
        ));
        let live=counter.budget().used_bytes();
        let root=Arc::downgrade(&host.root);
        if finish { assert!(host.finish().is_err()); }
        else { assert!(host.append(0,0,b"x").is_err()); }
        assert!(host.append(0,0,b"x").is_err()); assert!(host.finish().is_err());
        assert!(host.release_final_charges().is_err());
        drop(host);
        assert!(root.upgrade().is_some()); assert!(header.upgrade().is_some());
        assert!(directory.upgrade().is_some()); assert_eq!(counter.budget().used_bytes(),live);
        let host=RawHost {root:root.upgrade().unwrap()};
        host.retire().unwrap();
        assert!(header.upgrade().is_some()); assert!(directory.upgrade().is_some());
        assert_eq!(counter.budget().used_bytes(),live);
        host.release_final_charges().unwrap();
        assert!(header.upgrade().is_none()); assert!(directory.upgrade().is_none());
        assert_eq!(counter.budget().used_bytes(),OwnedBudget::resources().unwrap().initial_reserved_bytes);
        assert!(host.release_final_charges().is_err()); assert!(host.retire().is_err());
        drop(host); assert!(root.upgrade().is_none());
    }
    assert_eq!(snapshot_directory::OwnedSnapshotDirectory::factories_for_tests(),directories);
    assert_eq!(WRITER_FACTORIES.with(|n|n.get()),writers);
}

#[test]
fn incomplete_finish_latches_without_losing_completed_file_or_final_charges() {
    let fixture=Fixture::new(); let host=fixture.started();
    host.append_fixture_file(0).unwrap();
    let (file,header,counter)=host.inspect(|r|(
        r.files[0].original.clone(),r.header.identity.clone(),r.header.root.try_clone().unwrap(),
    ));
    let live=counter.budget().used_bytes();
    assert!(host.finish().is_err());
    assert!(host.finish().is_err());
    assert!(host.append_fixture_file(1).is_err());
    assert!(file.upgrade().is_some()); assert!(header.upgrade().is_some());
    assert_eq!(counter.budget().used_bytes(),live);
    host.retire().unwrap();
    assert!(file.upgrade().is_none()); assert!(header.upgrade().is_some());
    assert_eq!(counter.budget().used_bytes(),live);
    host.release_final_charges().unwrap();
    assert!(header.upgrade().is_none()); assert!(counter.budget().used_bytes()<live);
}

#[test]
fn failed_hash_and_external_handle_drop_do_not_retire_rooted_host() {
    let fixture=Fixture::new();
    let host=RawHost::prepare(fixture.header()).unwrap();
    host.start(snapshot_directory::RawSnapshotDirectory::parent_device().unwrap()).unwrap();
    host.append_fixture_file(0).unwrap();
    let (file,header,counter,path)=host.inspect(|r|(
        r.files[0].original.clone(),r.header.identity.clone(),r.header.root.try_clone().unwrap(),
        SnapshotDirectory::path(r.directory.as_ref()).to_path_buf(),
    ));
    let mut bad=host.inspect(|r|r.header.fixture.as_ref().unwrap().files[1].1.clone()); bad[0]^=1;
    let live=counter.budget().used_bytes();
    assert!(host.append(1,0,&bad).is_err()); // consuming core finish burns candidate
    host.inspect(|r| {assert!(r.writer.is_none()); assert_eq!(r.files.len(),1);});
    let factories=WRITER_FACTORIES.with(|n|n.get());
    assert!(host.append_fixture_file(1).is_err());
    assert!(host.start(snapshot_directory::RawSnapshotDirectory::parent_device().unwrap()).is_err());
    assert_eq!(WRITER_FACTORIES.with(|n|n.get()),factories);
    let root=Arc::downgrade(&host.root); drop(host);
    assert!(path.is_dir()); assert!(file.upgrade().is_some()); assert!(header.upgrade().is_some());
    assert_eq!(counter.budget().used_bytes(),live);
    // Test-only recovery of the exact original root, NOT a production constructor.
    let host=RawHost {root:root.upgrade().unwrap()};
    host.retire().unwrap(); assert!(!path.exists());
    assert!(header.upgrade().is_some()); assert_eq!(counter.budget().used_bytes(),live);
    host.release_final_charges().unwrap(); drop(host);
    assert!(root.upgrade().is_none()); assert!(header.upgrade().is_none());
}

#[test]
fn failed_start_is_sticky_and_keeps_pending_custody() {
    let fixture=Fixture::new(); let host=TestHost(RawHost::prepare(fixture.header()).unwrap());
    let actual=snapshot_directory::RawSnapshotDirectory::parent_device().unwrap();
    let factories=counts();
    assert!(host.start(actual^1).is_err());
    assert!(host.start(actual).is_err()); assert!(host.append(0,0,b"x").is_err());
    assert_eq!(counts(),factories);
    host.inspect(|r| {r.header.check().unwrap(); assert!(r.writer.is_none());});
    assert!(host.release_final_charges().is_err());
    host.retire().unwrap(); // no directory was created; explicit retirement still required
}

#[test]
fn failed_directory_retire_keeps_anchor_root_charges_and_refuses_retry() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let (directory,header,counter,path)=host.inspect(|r|(
        Arc::downgrade(&r.directory),r.header.identity.clone(),r.header.root.try_clone().unwrap(),
        SnapshotDirectory::path(r.directory.as_ref()).to_path_buf(),
    ));
    let blocker=path.join("synthetic-leftover"); std::fs::write(&blocker,b"leftover").unwrap();
    let live=counter.budget().used_bytes();
    assert!(host.retire().is_err());
    assert_eq!(directory.upgrade().unwrap().retention_for_tests(),(true,true,true));
    host.inspect(|r| {assert!(r.files.is_empty()); assert!(r.writer.is_none());});
    assert!(host.release_final_charges().is_err());
    std::fs::remove_file(blocker).unwrap(); // remove test blocker, NOT retry retirement
    assert!(host.retire().is_err());
    drop(host); // test helper also cannot retry or release this failed root
    assert!(path.is_dir()); assert!(header.upgrade().is_some());
    assert_eq!(counter.budget().used_bytes(),live);
    assert_eq!(directory.upgrade().unwrap().retention_for_tests(),(true,true,true));
    // Intentional failure retention lasts through process lifetime. Neither a
    // TempDir destructor nor an invented receipt is used to clean/refund it.
}

#[test]
fn d32_full_finite_geometry_and_scale_header_two_reads_ten_bytes_per_element() {
    let fixture=Fixture::d32(); let pending=fixture.header();
    let premise=pending.raw_premise.as_ref().unwrap();
    let head=semantic_weight(&pending,pllm_models::ModelOperator::OutputHead).unwrap();
    let head=premise.tensors.iter().find(|t|t.name==head).unwrap();
    let config_len=pending.fixture.as_ref().unwrap().files[0].1.len();
    let ledger=PassLedger::quote(&premise.files,&premise.tensors,config_len,head.end-head.begin).unwrap();
    let mut tensors:Vec<_>=premise.tensors.iter().map(|t|ExpectedTensor {
        name:t.name.clone(),file:t.file,dtype:t.dtype,shape:t.shape,rank:t.rank,
        begin:t.begin,end:t.end,scale:t.scale.clone(),scale_index:None,
    }).collect();
    let without=PassLedger::quote(&premise.files,&tensors,config_len,head.end-head.begin).unwrap();
    assert_eq!(ledger.reads-without.reads,2*16384);
    assert_eq!(ledger.read_bytes-without.read_bytes,10*16384);
    assert_eq!(ledger.decodes-without.decodes,3*16384);
    tensors.clear();
    let host=TestHost(RawHost::prepare(pending).unwrap());
    host.start(snapshot_directory::RawSnapshotDirectory::parent_device().unwrap()).unwrap();
    host.ingest_fixture(4096).unwrap(); host.finish().unwrap();
}

#[test]
fn d32_invalid_scale_or_nonfinite_unused_element_is_terminal() {
    for scale in [false,true] {
        let mut fixture=Fixture::d32();
        let name=if scale {"zz_unused.weight_scale_inv"} else {"zz_unused.weight"};
        let tensor=fixture.tensors.iter().find(|t|t.name==name).unwrap();
        let offset=fixture.data()+tensor.begin as usize;
        if scale { fixture.files[2].1[offset..offset+2].copy_from_slice(&0u16.to_le_bytes()); }
        else { fixture.files[2].1[offset]=0x7f; }
        fixture.repin(2);
        let host=fixture.collect(4096).unwrap();
        assert!(host.finish().is_err()); assert!(host.finish().is_err());
        host.inspect(|r|assert_eq!(r.files.len(),3));
    }
}

// Stage 1 genuine raw-reader custody. Definitions only; not execution evidence.
fn lookup_index(host: &RawHost) -> usize {
    host.inspect(|r| r.header.original.plan().prefill.operations.iter()
        .position(|op| op.operator == pllm_models::ModelOperator::TokenLookup).unwrap())
}

#[test]
fn reader_requires_seal_and_claims_once_without_ingestion_reclassification() {
    let fixture=Fixture::new(); let pending=TestHost(RawHost::prepare(fixture.header()).unwrap());
    assert!(pending.claim_reader().is_err());
    assert!(!pending.root.state.lock().unwrap().reader_claimed);
    pending.start(snapshot_directory::RawSnapshotDirectory::parent_device().unwrap()).unwrap();
    pending.ingest_fixture(17).unwrap(); let host=pending; host.finish().unwrap();
    let reader=host.claim_reader().unwrap();
    assert!(host.claim_reader().is_err());
    assert!(host.root.state.lock().unwrap().stage==HostStage::Sealed);
    drop(reader);
    assert!(host.claim_reader().is_err()); // no reissue after last descendant
    host.retire().unwrap(); host.release_final_charges().unwrap();
}

#[test]
fn semantic_window_uses_original_operation_extent_and_pins_retirement() {
    let fixture=Fixture::new(); let host=fixture.collect(19).unwrap(); host.finish().unwrap();
    let operation=lookup_index(&host);
    let reader=host.claim_reader().unwrap();
    let tensor=reader.weight(RawGraph::Prefill,operation).unwrap();
    assert_eq!(tensor.shape(),[3,2]);
    let window=tensor.read(2,4).unwrap();
    assert_eq!(window.bytes(),[0x00,0x40,0x80,0x40]);
    let (file,counter)=host.inspect(|r|(r.files[2].original.clone(),r.header.root.try_clone().unwrap()));
    let live=counter.budget().used_bytes();
    assert!(host.retire().is_err());
    assert!(!host.root.state.lock().unwrap().retire_attempted);
    assert!(file.upgrade().is_some()); assert_eq!(counter.budget().used_bytes(),live);
    assert!(host.release_final_charges().is_err());
    drop(window); drop(tensor); // Rust borrow keeps reader alive through both
    assert!(host.retire().is_err());
    drop(reader);
    host.retire().unwrap(); assert!(file.upgrade().is_none());
    host.release_final_charges().unwrap();
}

#[test]
fn semantic_reader_survives_caller_drop_without_losing_retention_root() {
    let fixture=Fixture::new(); let host=fixture.collect(23).unwrap(); host.finish().unwrap();
    let operation=lookup_index(&host);
    let root=Arc::downgrade(&host.root);
    let reader=host.claim_reader().unwrap(); drop(host);
    let tensor=reader.weight(RawGraph::Prefill,operation).unwrap();
    assert_eq!(tensor.read(0,2).unwrap().bytes(),[0x80,0x3f]);
    let host=RawHost {root:root.upgrade().unwrap()};
    assert!(host.retire().is_err());
    drop(tensor); drop(reader); drop(host);
    // Reader Drop never performs cleanup or releases original host charges.
    let host=RawHost {root:root.upgrade().unwrap()};
    host.retire().unwrap(); host.release_final_charges().unwrap(); drop(host);
    assert!(root.upgrade().is_none());
}

#[test]
fn matching_bytes_and_labels_do_not_replace_original_backing_after_seal() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let operation=lookup_index(&host);
    let reader=host.claim_reader().unwrap();
    let tensor=reader.weight(RawGraph::Prefill,operation).unwrap();
    host.inspect(|r| {
        let fixture=r.header.fixture.as_ref().unwrap();
        let mut writer=PublicFileWriter::new_in(fixture.sizes[2],fixture.hashes[2],
            Arc::clone(&r.directory) as Arc<dyn SnapshotDirectory>).unwrap();
        writer.write(&fixture.files[2].1).unwrap();
        r.files[2].file=Some(Arc::new(writer.finish().unwrap()));
    });
    assert!(tensor.read(0,2).is_err());
    assert!(tensor.read(0,2).is_err()); // late failure terminal, no partial window
}

#[test]
fn substituted_equal_header_and_foreign_operation_cannot_supply_semantics() {
    for header_swap in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let operation=lookup_index(&host);
        let reader=host.claim_reader().unwrap();
        if header_swap {
            host.inspect(|r| r.header=Fixture::new().header());
            assert!(reader.weight(RawGraph::Prefill,operation).is_err());
        } else {
            assert!(reader.weight(RawGraph::Prefill,usize::MAX).is_err());
            assert!(reader.weight(RawGraph::Prefill,operation).is_err());
        }
    }
}

#[test]
fn window_rejects_empty_oversize_and_last_byte_overrun_without_publication() {
    for (offset,len) in [(0,0),(0,FRAME+1),(11,2),(u64::MAX,2)] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let reader=host.claim_reader().unwrap();
        let tensor=reader.weight(RawGraph::Prefill,lookup_index(&host)).unwrap();
        assert!(tensor.read(offset,len).is_err());
        assert!(tensor.read(0,2).is_err());
    }
}

#[test]
fn reader_claim_exact_byte_boundary_precedes_factory_and_consumes_attempt() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let mut outside=host.inspect(|r|r.header.root.charge_external(0).unwrap());
        let required=independent_reader_bytes()+size_of::<OwnedExternalCharge>();
        let used=outside.budget().used_bytes();
        outside.resize_payload(MAX_PROGRAM_BYTES-used-required+usize::from(short)).unwrap();
        let before=READER_FACTORIES.with(|n|n.get());
        let result=host.claim_reader();
        assert_eq!(result.is_err(),short);
        assert_eq!(READER_FACTORIES.with(|n|n.get()),before+usize::from(!short));
        assert!(host.claim_reader().is_err());
        if !short { assert_eq!(outside.budget().used_bytes(),MAX_PROGRAM_BYTES); }
        drop(result); drop(outside);
    }
}

#[test]
fn reader_claim_exact_work_boundary_shares_original_ledger_before_factory() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let sibling=host.inspect(|r| {
            let h=&r.header;
            let bootstrap=h.original.lower_bootstrap_bytes(h.limits).unwrap();
            let sibling=h.original.lower(h.root.charge_external(bootstrap).unwrap(),h.original.workload(),h.limits).unwrap();
            let fill=h.limits.max_work-sibling.resources().prepaid_work-(1+READER_CLAIM_WORK)+u64::from(short);
            h.original.prepay_compiler_metadata(&h.controller,fill).unwrap();
            sibling
        });
        let before=READER_FACTORIES.with(|n|n.get());
        let result=host.claim_reader();
        assert_eq!(result.is_err(),short);
        assert_eq!(READER_FACTORIES.with(|n|n.get()),before+usize::from(!short));
        host.inspect(|r| {
            assert_eq!(sibling.prepay_primitive_work(&r.header.controller,0).is_err(),short);
            if !short { assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err()); }
        });
        drop(result);
    }
}

#[test]
fn retained_windows_pay_overlap_and_wipe_before_scratch_refund() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let reader=host.claim_reader().unwrap();
    let tensor=reader.weight(RawGraph::Prefill,lookup_index(&host)).unwrap();
    let counter=host.inspect(|r|r.header.root.try_clone().unwrap());
    let baseline=counter.budget().used_bytes();
    let first=tensor.read(0,2).unwrap();
    let one=counter.budget().used_bytes();
    let second=tensor.read(2,2).unwrap();
    let both=counter.budget().used_bytes();
    assert_eq!(one-baseline,independent_window_bytes()+size_of::<OwnedExternalCharge>());
    assert_eq!(both-one,one-baseline);
    assert_eq!(first.bytes(),[0x80,0x3f]); assert_eq!(second.bytes(),[0,0x40]);
    WINDOW_DROP.with(|s|s.set(None));
    drop(second);
    assert_eq!(WINDOW_DROP.with(|s|s.get()),Some((true,both)));
    assert_eq!(counter.budget().used_bytes(),one);
    drop(first);
    assert_eq!(WINDOW_DROP.with(|s|s.get()),Some((true,one)));
    assert_eq!(counter.budget().used_bytes(),baseline);
}

#[test]
fn window_exact_byte_boundary_admits_old_new_overlap_before_factory() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let reader=host.claim_reader().unwrap();
        let tensor=reader.weight(RawGraph::Prefill,lookup_index(&host)).unwrap();
        let first=tensor.read(0,2).unwrap();
        let mut outside=host.inspect(|r|r.header.root.charge_external(0).unwrap());
        let required=independent_window_bytes()+size_of::<OwnedExternalCharge>();
        let used=outside.budget().used_bytes();
        outside.resize_payload(MAX_PROGRAM_BYTES-used-required+usize::from(short)).unwrap();
        let before=WINDOW_FACTORIES.with(|n|n.get());
        let result=tensor.read(2,2);
        assert_eq!(result.is_err(),short);
        assert_eq!(WINDOW_FACTORIES.with(|n|n.get()),before+usize::from(!short));
        assert_eq!(first.bytes(),[0x80,0x3f]);
        if !short { assert_eq!(outside.budget().used_bytes(),MAX_PROGRAM_BYTES); }
        drop(result); drop(outside);
    }
}

#[test]
fn last_window_work_short_burns_same_ledger_and_publishes_no_window() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let reader=host.claim_reader().unwrap();
        let tensor=reader.weight(RawGraph::Prefill,lookup_index(&host)).unwrap();
        let first=tensor.read(0,10).unwrap();
        let sibling=host.inspect(|r| {
            let h=&r.header;
            let bootstrap=h.original.lower_bootstrap_bytes(h.limits).unwrap();
            let sibling=h.original.lower(h.root.charge_external(bootstrap).unwrap(),h.original.workload(),h.limits).unwrap();
            let fill=h.limits.max_work-sibling.resources().prepaid_work-(1+RAW_WINDOW_WORK)+u64::from(short);
            h.original.prepay_primitive_work(&h.controller,fill).unwrap();
            sibling
        });
        let before=WINDOW_FACTORIES.with(|n|n.get());
        let last=tensor.read(10,2);
        assert_eq!(last.is_err(),short);
        assert_eq!(WINDOW_FACTORIES.with(|n|n.get()),before+usize::from(!short));
        assert_eq!(first.bytes().len(),10);
        drop(last); // no logical work refund after a successful final window
        host.inspect(|r| {
            assert_eq!(sibling.prepay_primitive_work(&r.header.controller,0).is_err(),short);
            if !short { assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err()); }
        });
    }
}

#[test]
fn busy_retire_does_not_start_cleanup_but_actual_cleanup_failure_stays_sticky() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let reader=host.claim_reader().unwrap();
    let (directory,counter,path)=host.inspect(|r|(
        Arc::downgrade(&r.directory),r.header.root.try_clone().unwrap(),
        SnapshotDirectory::path(r.directory.as_ref()).to_path_buf(),
    ));
    let blocker=path.join("stage-one-cleanup-blocker"); std::fs::write(&blocker,b"leftover").unwrap();
    let live=counter.budget().used_bytes();
    assert!(host.retire().is_err());
    assert_eq!(directory.upgrade().unwrap().retention_for_tests(),(true,true,false));
    assert!(!host.root.state.lock().unwrap().retire_attempted);
    drop(reader);
    assert_eq!(counter.budget().used_bytes(),live); // reader controls remain rooted
    assert!(host.retire().is_err());
    assert!(host.root.state.lock().unwrap().retire_attempted);
    assert_eq!(directory.upgrade().unwrap().retention_for_tests(),(true,true,true));
    std::fs::remove_file(blocker).unwrap();
    assert!(host.retire().is_err()); assert!(host.release_final_charges().is_err());
    assert_eq!(counter.budget().used_bytes(),live);
}

// Stage 2 definitions: genuine sealed raw host -> original owned token request
// -> exact BF16 lookup. No fixture can construct the activation directly.
#[test]
fn genuine_lookup_owns_complete_ordered_immutable_tokens_and_all_rows() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let mut tokens=vec![2,0,1,2];
    let input=host.claim_reader().unwrap().admit_tokens(RawGraph::Prefill,&tokens).unwrap();
    tokens.fill(u64::MAX); drop(tokens);
    let output=PrivateSourceInstance::new(input).token_lookup().unwrap();
    assert_eq!(output.shape(),[1,4,2]);
    assert_eq!(output.tokens_for_tests(),[2,0,1,2]);
    assert_eq!(output.words(),[0x3f00,0x3e80,0x3f80,0x4000,0x4080,0x4100,0x3f00,0x3e80]);
    output.check_source(&host).unwrap();
    assert!(host.claim_reader().is_err());
    assert!(host.retire().is_err());
    assert!(!host.root.state.lock().unwrap().retire_attempted);
    drop(output);
    host.retire().unwrap(); host.release_final_charges().unwrap();
}

#[test]
fn equal_raw_bytes_labels_and_tokens_do_not_substitute_foreign_original_source() {
    let fixture=Fixture::new();
    let a=fixture.collect(17).unwrap(); a.finish().unwrap();
    let b=Fixture::new().collect(19).unwrap(); b.finish().unwrap();
    let make=|host:&RawHost| PrivateSourceInstance::new(host.claim_reader().unwrap()
        .admit_tokens(RawGraph::Prefill,&[0,1,2,0]).unwrap()).token_lookup().unwrap();
    let left=make(&a); let right=make(&b);
    assert_eq!(left.words(),right.words());
    assert_eq!(left.tokens_for_tests(),right.tokens_for_tests());
    left.check_source(&a).unwrap(); right.check_source(&b).unwrap();
    assert!(left.check_source(&b).is_err()); assert!(right.check_source(&a).is_err());
    // Neither request nor instance nor activation is Clone. No constructor takes
    // an existing activation/slice plus a replacement request, label or source.
    assert!(a.claim_reader().is_err()); assert!(b.claim_reader().is_err());
}

#[test]
fn lookup_output_survives_all_callers_drop_and_preserves_retirement_retry() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let root=Arc::downgrade(&host.root);
    let output=PrivateSourceInstance::new(host.claim_reader().unwrap()
        .admit_tokens(RawGraph::Prefill,&[0,1,2,0]).unwrap()).token_lookup().unwrap();
    drop(host);
    assert_eq!(output.words()[..2],[0x3f80,0x4000]);
    let host=RawHost {root:root.upgrade().unwrap()};
    assert!(host.retire().is_err()); assert!(host.release_final_charges().is_err());
    drop(output);
    host.retire().unwrap(); host.release_final_charges().unwrap(); drop(host);
    assert!(root.upgrade().is_none());
}

#[test]
fn original_prefill_and_decode_shapes_reject_flattening_and_out_of_domain_ids() {
    for tokens in [vec![],vec![0],vec![0,1,2],vec![0;8],vec![0,1,2,3],vec![0,1,2,u64::MAX]] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let before=raw_clear_producer::TOKEN_FACTORIES.with(|n|n.get());
        assert!(host.claim_reader().unwrap().admit_tokens(RawGraph::Prefill,&tokens).is_err());
        assert_eq!(raw_clear_producer::TOKEN_FACTORIES.with(|n|n.get()),before);
        assert!(host.claim_reader().is_err());
        assert!(!host.root.state.lock().unwrap().reader_live);
    }
    let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
    let output=PrivateSourceInstance::new(host.claim_reader().unwrap()
        .admit_tokens(RawGraph::Decode,&[1]).unwrap()).token_lookup().unwrap();
    assert_eq!(output.shape(),[1,1,2]); assert_eq!(output.words(),[0x4080,0x4100]);
}

#[test]
fn late_final_lookup_row_failure_wipes_partial_output_and_consumes_request() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let input=host.claim_reader().unwrap().admit_tokens(RawGraph::Prefill,&[0,1,2,0]).unwrap();
    let (_,work)=raw_clear_producer::lookup_quote_for_tests(&input);
    let sibling=leave_reader_work(&host,work,false); // EXACT final-fault work, no spare unit
    let counter=host.inspect(|r|r.header.root.try_clone().unwrap());
    let live=counter.budget().used_bytes();
    raw_clear_producer::LOOKUP_DROP.with(|n|n.set(None));
    raw_clear_producer::FAIL_ROW.with(|n|n.set(Some(3)));
    let result=PrivateSourceInstance::new(input).token_lookup();
    raw_clear_producer::FAIL_ROW.with(|n|n.set(None));
    assert!(result.is_err());
    let (wiped,covered)=raw_clear_producer::LOOKUP_DROP.with(|n|n.get()).unwrap();
    assert!(wiped); assert!(covered>live);
    assert!(counter.budget().used_bytes()<live); // request/output gone, original host retained
    assert!(host.claim_reader().is_err());
    assert!(!host.root.state.lock().unwrap().reader_live);
    host.inspect(|r| {
        sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
        assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());
    });
    host.retire().unwrap(); host.release_final_charges().unwrap();
}

#[test]
fn lookup_rechecks_exact_original_backing_after_request_admission() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let input=host.claim_reader().unwrap().admit_tokens(RawGraph::Prefill,&[0,1,2,0]).unwrap();
    host.inspect(|r| {
        let fixture=r.header.fixture.as_ref().unwrap();
        let mut writer=PublicFileWriter::new_in(fixture.sizes[2],fixture.hashes[2],
            Arc::clone(&r.directory) as Arc<dyn SnapshotDirectory>).unwrap();
        writer.write(&fixture.files[2].1).unwrap();
        r.files[2].file=Some(Arc::new(writer.finish().unwrap()));
    });
    let before=raw_clear_producer::LOOKUP_FACTORIES.with(|n|n.get());
    assert!(PrivateSourceInstance::new(input).token_lookup().is_err());
    assert_eq!(raw_clear_producer::LOOKUP_FACTORIES.with(|n|n.get()),before);
}

// All existing reservation/norm quotes exclude the new separate bootstrap.
fn leave_primitive_work(host: &RawHost, work: u64, short: bool) -> OriginalSourcePlan {
    leave_reader_work(host,work.checked_add(24).unwrap(),short)
}
fn leave_reader_work(host: &RawHost, work: u64, short: bool) -> OriginalSourcePlan {
    host.inspect(|r| {
        let h=&r.header;
        let bootstrap=h.original.lower_bootstrap_bytes(h.limits).unwrap();
        let sibling=h.original.lower(h.root.charge_external(bootstrap).unwrap(),h.original.workload(),h.limits).unwrap();
        let fill=h.limits.max_work-sibling.resources().prepaid_work-work+u64::from(short);
        h.original.prepay_primitive_work(&h.controller,fill).unwrap();
        sibling
    })
}

#[test]
fn request_token_capacity_exact_and_one_byte_short_before_copy_factory() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let reader=host.claim_reader().unwrap();
        let (bytes,_)=raw_clear_producer::input_quote_for_tests(&host,RawGraph::Prefill,4);
        let mut outside=host.inspect(|r|r.header.root.charge_external(0).unwrap());
        let used=outside.budget().used_bytes();
        outside.resize_payload(MAX_PROGRAM_BYTES-used-bytes-size_of::<OwnedExternalCharge>()+usize::from(short)).unwrap();
        let before=raw_clear_producer::TOKEN_FACTORIES.with(|n|n.get());
        let input=reader.admit_tokens(RawGraph::Prefill,&[0,1,2,0]);
        assert_eq!(input.is_err(),short);
        assert_eq!(raw_clear_producer::TOKEN_FACTORIES.with(|n|n.get()),before+usize::from(!short));
        if !short { assert_eq!(outside.budget().used_bytes(),MAX_PROGRAM_BYTES); }
        assert!(host.claim_reader().is_err());
        drop(input); drop(outside);
    }
}

#[test]
fn request_work_exact_and_one_short_debit_original_ledger_before_factory() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let reader=host.claim_reader().unwrap();
        let (_,work)=raw_clear_producer::input_quote_for_tests(&host,RawGraph::Prefill,4);
        let sibling=leave_reader_work(&host,work,short);
        let before=raw_clear_producer::TOKEN_FACTORIES.with(|n|n.get());
        let result=reader.admit_tokens(RawGraph::Prefill,&[0,1,2,0]);
        assert_eq!(result.is_err(),short);
        assert_eq!(raw_clear_producer::TOKEN_FACTORIES.with(|n|n.get()),before+usize::from(!short));
        drop(result);
        host.inspect(|r| {
            assert_eq!(sibling.prepay_primitive_work(&r.header.controller,0).is_err(),short);
            if !short { assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err()); }
        });
    }
}

#[test]
fn lookup_capacity_exact_and_one_byte_short_before_any_output_factory() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let input=host.claim_reader().unwrap().admit_tokens(RawGraph::Prefill,&[0,1,2,0]).unwrap();
        let (bytes,_)=raw_clear_producer::lookup_quote_for_tests(&input);
        let mut outside=host.inspect(|r|r.header.root.charge_external(0).unwrap());
        let used=outside.budget().used_bytes();
        outside.resize_payload(MAX_PROGRAM_BYTES-used-bytes-size_of::<OwnedExternalCharge>()+usize::from(short)).unwrap();
        let before=raw_clear_producer::LOOKUP_FACTORIES.with(|n|n.get());
        let output=PrivateSourceInstance::new(input).token_lookup();
        assert_eq!(output.is_err(),short);
        assert_eq!(raw_clear_producer::LOOKUP_FACTORIES.with(|n|n.get()),before+usize::from(!short));
        if !short { assert_eq!(outside.budget().used_bytes(),MAX_PROGRAM_BYTES); }
        drop(output); drop(outside);
    }
}

#[test]
fn lookup_work_exact_and_one_short_prepay_all_rows_before_output_factory() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let input=host.claim_reader().unwrap().admit_tokens(RawGraph::Prefill,&[0,1,2,0]).unwrap();
        let (_,work)=raw_clear_producer::lookup_quote_for_tests(&input);
        let sibling=leave_reader_work(&host,work,short);
        let before=raw_clear_producer::LOOKUP_FACTORIES.with(|n|n.get());
        let output=PrivateSourceInstance::new(input).token_lookup();
        assert_eq!(output.is_err(),short);
        assert_eq!(raw_clear_producer::LOOKUP_FACTORIES.with(|n|n.get()),before+usize::from(!short));
        drop(output);
        host.inspect(|r| {
            assert_eq!(sibling.prepay_primitive_work(&r.header.controller,0).is_err(),short);
            if !short { assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err()); }
        });
    }
}

#[test]
fn forged_operation_or_equal_foreign_request_reader_cannot_publish_lookup() {
    for foreign in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let other=Fixture::new().collect(19).unwrap(); other.finish().unwrap();
        let mut input=host.claim_reader().unwrap().admit_tokens(RawGraph::Prefill,&[0,1,2,0]).unwrap();
        if foreign {
            raw_clear_producer::replace_reader_for_tests(&mut input,other.claim_reader().unwrap());
        } else {
            raw_clear_producer::replace_operation_for_tests(&mut input,usize::MAX);
        }
        let before=raw_clear_producer::LOOKUP_FACTORIES.with(|n|n.get());
        assert!(PrivateSourceInstance::new(input).token_lookup().is_err());
        assert_eq!(raw_clear_producer::LOOKUP_FACTORIES.with(|n|n.get()),before);
        assert!(host.claim_reader().is_err());
    }
}

#[test]
fn exact_bf16_lookup_preserves_signed_zero_subnormal_and_wipes_before_refund() {
    let mut fixture=Fixture::new();
    let start=fixture.data();
    // Repin BOTH original head and embedding in this synthetic source. Genuine
    // registration still checks complete inventory, finite/Q8 contract and tie.
    for tensor in &fixture.tensors {
        if tensor.name=="model.embed_tokens.weight" || tensor.name=="lm_head.weight" {
            let begin=start+tensor.begin as usize;
            fixture.files[2].1[begin..begin+4].copy_from_slice(&[0x00,0x80,0x01,0x00]);
        }
    }
    fixture.repin(2);
    let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let counter=host.inspect(|r|r.header.root.try_clone().unwrap());
    let baseline=counter.budget().used_bytes();
    let output=PrivateSourceInstance::new(host.claim_reader().unwrap()
        .admit_tokens(RawGraph::Prefill,&[0,0,0,0]).unwrap()).token_lookup().unwrap();
    assert_eq!(output.words(),[0x8000,0x0001,0x8000,0x0001,0x8000,0x0001,0x8000,0x0001]);
    let live=counter.budget().used_bytes();
    raw_clear_producer::LOOKUP_DROP.with(|n|n.set(None));
    raw_clear_producer::TOKEN_DROP.with(|n|n.set(None));
    drop(output);
    assert_eq!(raw_clear_producer::LOOKUP_DROP.with(|n|n.get()),Some((true,live)));
    let (wiped,covered)=raw_clear_producer::TOKEN_DROP.with(|n|n.get()).unwrap();
    assert!(wiped); assert!(covered<live); assert!(covered>baseline);
    // Reader control charge, header and disk pins survive until explicit final release.
    assert!(counter.budget().used_bytes()>baseline);
    host.retire().unwrap(); host.release_final_charges().unwrap();
}

// Native retirement reservation regressions. Definitions only, UNRUN.
#[test]
fn begin_retire_claim_first_busy_preserves_exact_custody_then_retries() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let reader=host.claim_reader().unwrap();
    let tensor=reader.weight(RawGraph::Prefill,lookup_index(&host)).unwrap();
    let window=tensor.read(0,2).unwrap();
    let snapshot=|| {
        let state=host.root.state.lock().unwrap();
        let payload=state.payload.as_ref().unwrap();
        (state.stage,state.retire_reserved,state.retire_attempted,state.reader_live,
            state.reader_claimed,state.reader_failed,state.retained.as_ref().map(Arc::as_ptr),
            payload.files.len(),payload.writer.is_some(),payload.header.root.budget().used_bytes(),
            payload.directory.retention_for_tests(),Arc::strong_count(&host.root))
    };
    let before=snapshot();
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
    assert!(before==snapshot());
    assert_eq!(window.bytes(),[0x80,0x3f]);
    drop(window); drop(tensor);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
    drop(reader);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    assert!(host.release_final_charges().is_err());
    host.retire().unwrap(); host.release_final_charges().unwrap();
}

#[test]
fn begin_retire_reservation_first_blocks_claim_ingestion_reads_and_second_begin() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let (file,counter)=host.inspect(|r|(r.files[2].original.clone(),r.header.root.try_clone().unwrap()));
    let covered=counter.budget().used_bytes();
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    assert!(host.claim_reader().is_err());
    assert!(host.start(0).is_err());
    assert!(host.append(0,0,&[1]).is_err());
    assert!(host.finish().is_err());
    assert!(host.begin_retire().is_err());
    assert!(host.release_final_charges().is_err());
    {
        let state=host.root.state.lock().unwrap();
        assert!(state.retire_reserved); assert!(!state.retire_attempted);
        assert!(state.stage==HostStage::Failed); assert!(reader_payload(&state).is_err());
        assert!(Arc::ptr_eq(state.retained.as_ref().unwrap(),&host.root));
    }
    assert!(file.upgrade().is_some()); assert_eq!(counter.budget().used_bytes(),covered);
    host.retire().unwrap(); assert!(file.upgrade().is_none());
    assert!(!host.root.state.lock().unwrap().retire_reserved);
    assert!(host.begin_retire().is_err());
    host.retire().unwrap(); // historical direct-native successful-retire idempotence
    host.release_final_charges().unwrap();
    assert!(host.begin_retire().is_err()); assert!(host.retire().is_err());
}

#[test]
fn begin_retire_pending_reservation_roots_caller_drop_until_explicit_release() {
    let fixture=Fixture::new(); let host=RawHost::prepare(fixture.header()).unwrap();
    let root=Arc::downgrade(&host.root);
    let counter=host.inspect(|r|r.header.root.try_clone().unwrap());
    let covered=counter.budget().used_bytes();
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    assert!(host.start(0).is_err()); drop(host);
    let host=RawHost {root:root.upgrade().unwrap()};
    assert_eq!(counter.budget().used_bytes(),covered);
    assert!(host.release_final_charges().is_err());
    host.retire().unwrap(); assert_eq!(counter.budget().used_bytes(),covered);
    host.release_final_charges().unwrap(); drop(host);
    assert!(root.upgrade().is_none());
}

#[test]
fn begin_retire_bounded_claim_reservation_race_has_exactly_one_winner() {
    for _ in 0..8 {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let start=std::sync::Barrier::new(3);
        let completed=std::sync::Barrier::new(3);
        let (reader,readiness)=std::thread::scope(|scope| {
            let claiming=scope.spawn(|| {
                start.wait();
                let reader=host.claim_reader();
                completed.wait(); // keep successful reader live through both attempts
                reader
            });
            let retiring=scope.spawn(|| {
                start.wait();
                let readiness=host.begin_retire();
                completed.wait();
                readiness
            });
            start.wait(); completed.wait();
            (claiming.join().unwrap(),retiring.join().unwrap())
        });
        let readiness=readiness.unwrap();
        assert!(matches!((&reader,readiness),
            (Ok(_),RetirementReadiness::Busy)|(Err(_),RetirementReadiness::Ready)));
        drop(reader);
        if readiness==RetirementReadiness::Busy {
            assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
        }
        assert!(host.claim_reader().is_err());
        host.retire().unwrap(); host.release_final_charges().unwrap();
    }
}

#[test]
fn begin_retire_consumed_cleanup_failure_remains_sticky_and_rooted() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let (directory,counter,path)=host.inspect(|r|(
        Arc::downgrade(&r.directory),r.header.root.try_clone().unwrap(),
        SnapshotDirectory::path(r.directory.as_ref()).to_path_buf(),
    ));
    let blocker=path.join("reserved-cleanup-blocker"); std::fs::write(&blocker,b"leftover").unwrap();
    let covered=counter.budget().used_bytes();
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    assert!(host.retire().is_err());
    {
        let state=host.root.state.lock().unwrap();
        assert!(!state.retire_reserved); assert!(state.retire_attempted);
        assert!(Arc::ptr_eq(state.retained.as_ref().unwrap(),&host.root));
    }
    assert_eq!(directory.upgrade().unwrap().retention_for_tests(),(true,true,true));
    std::fs::remove_file(blocker).unwrap();
    assert!(host.begin_retire().is_err()); assert!(host.retire().is_err());
    assert!(host.release_final_charges().is_err()); assert!(host.claim_reader().is_err());
    assert_eq!(counter.budget().used_bytes(),covered);
}

// OriginalReservation slice: genuine owners and original admission, UNRUN.
fn genuine_reservation_input(host: &RawHost) -> RawLookupActivation {
    PrivateSourceInstance::new(host.claim_reader().unwrap()
        .admit_tokens(RawGraph::Prefill,&[0,1,2,0]).unwrap()).token_lookup().unwrap()
}
fn consuming_original_reservation(output: RawLookupActivation, demand: PrimitiveQuote)
    -> Result<raw_clear_producer::OriginalReservation<RawLookupActivation>,Error> {
    // Returned owner has NO borrow into this local output or any local buffer.
    output.reserve_original(demand)
}

#[test]
fn original_reservation_returned_owner_retains_genuine_reader_and_all_tokens() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let root=Arc::downgrade(&host.root);
    let reserved=consuming_original_reservation(genuine_reservation_input(&host),
        PrimitiveQuote::checked(64,0).unwrap()).unwrap();
    drop(host);
    let host=RawHost {root:root.upgrade().unwrap()};
    assert_eq!(reserved.tokens_for_tests(),[0,1,2,0]);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
    assert!(host.claim_reader().is_err());
    drop(reserved);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    host.retire().unwrap(); host.release_final_charges().unwrap(); drop(host);
    assert!(root.upgrade().is_none());
}

#[test]
fn original_reservation_exact_bytes_and_one_short_precede_destination_factory() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let output=genuine_reservation_input(&host);
        let quote=raw_clear_producer::reservation_quote_for_tests(&output,raw_clear_producer::reserved_test_demand());
        let mut outside=host.inspect(|r|r.header.root.charge_external(0).unwrap());
        let used=outside.budget().used_bytes();
        outside.resize_payload(MAX_PROGRAM_BYTES-used-quote.resident_bytes-size_of::<OwnedExternalCharge>()+usize::from(short)).unwrap();
        let before=raw_clear_producer::RESERVED_FACTORIES.with(|n|n.get());
        let result=raw_clear_producer::reserved_test_factory(output,false);
        assert_eq!(result.is_err(),short);
        assert_eq!(raw_clear_producer::RESERVED_FACTORIES.with(|n|n.get()),before+usize::from(!short));
        if let Ok(owner)=&result {
            assert_eq!(outside.budget().used_bytes(),MAX_PROGRAM_BYTES);
            let (shape,words,capacity)=owner.retained_for_tests();
            assert_eq!(shape,[1,4,2]); assert_eq!(capacity,32);
            assert_eq!(words,[0x3f80,0x4000,0x4080,0x4100,0x3f00,0x3e80,0x3f80,0x4000]);
            assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
        }
        drop(result); drop(outside);
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
        host.retire().unwrap(); host.release_final_charges().unwrap();
    }
}

#[test]
fn original_reservation_exact_work_and_one_short_precede_destination_factory() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let output=genuine_reservation_input(&host);
        let quote=raw_clear_producer::reservation_quote_for_tests(&output,raw_clear_producer::reserved_test_demand());
        let sibling=leave_primitive_work(&host,quote.work,short);
        let before=raw_clear_producer::RESERVED_FACTORIES.with(|n|n.get());
        let result=raw_clear_producer::reserved_test_factory(output,false);
        assert_eq!(result.is_err(),short);
        assert_eq!(raw_clear_producer::RESERVED_FACTORIES.with(|n|n.get()),before+usize::from(!short));
        host.inspect(|r| {
            assert_eq!(sibling.prepay_primitive_work(&r.header.controller,0).is_err(),short);
            if !short { assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err()); }
        });
        drop(result);
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
        host.retire().unwrap(); host.release_final_charges().unwrap();
    }
}

#[test]
fn original_reservation_and_real_sibling_phase_share_cumulative_original_meter() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let output=genuine_reservation_input(&host);
    let quote=raw_clear_producer::reservation_quote_for_tests(&output,raw_clear_producer::reserved_test_demand());
    let sibling=leave_primitive_work(&host,quote.work.checked_add(2).unwrap(),false);
    host.inspect(|r|sibling.prepay_primitive_work(&r.header.controller,1).unwrap());
    let result=raw_clear_producer::reserved_test_factory(output,false).unwrap();
    host.inspect(|r| {
        sibling.prepay_primitive_work(&r.header.controller,1).unwrap();
        assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());
    });
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
    drop(result);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    host.retire().unwrap(); host.release_final_charges().unwrap();
}

#[test]
fn original_reservation_late_factory_error_wipes_full_capacity_and_spends_exact_work() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let output=genuine_reservation_input(&host);
    let quote=raw_clear_producer::reservation_quote_for_tests(&output,raw_clear_producer::reserved_test_demand());
    let sibling=leave_primitive_work(&host,quote.work,false);
    let counter=host.inspect(|r|r.header.root.try_clone().unwrap());
    let live=counter.budget().used_bytes();
    let before=raw_clear_producer::RESERVED_FACTORIES.with(|n|n.get());
    raw_clear_producer::RESERVED_DROP.with(|n|n.set(None));
    assert!(raw_clear_producer::reserved_test_factory(output,true).is_err());
    assert_eq!(raw_clear_producer::RESERVED_FACTORIES.with(|n|n.get()),before+1);
    let (wiped_freed_covered,covered)=raw_clear_producer::RESERVED_DROP.with(|n|n.get()).unwrap();
    assert!(wiped_freed_covered); assert!(covered>live);
    assert!(counter.budget().used_bytes()<live);
    host.inspect(|r| {
        sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
        assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());
    });
    // Consumed producer was not returned and no source/request can replay it.
    assert!(host.claim_reader().is_err());
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    host.retire().unwrap(); host.release_final_charges().unwrap();
}

#[test]
fn original_reservation_refuses_forged_key_and_valid_but_different_original_operation() {
    for case in 0..4 {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let output=genuine_reservation_input(&host);
        let (graph,operation,tensor)=host.inspect(|r| {
            let graph=if case==2 {RawGraph::Decode} else {RawGraph::Prefill};
            let plan=r.header.original.plan();
            let operations=match graph {RawGraph::Prefill=>&plan.prefill.operations,RawGraph::Decode=>&plan.decode.operations};
            let kind=if case==3 {pllm_models::ModelOperator::Linear} else {pllm_models::ModelOperator::TokenLookup};
            let op=operations.iter().position(|op|op.operator==kind).unwrap();
            let name=original_weight(r,graph,op).unwrap();
            let tensor=r.header.raw_premise.as_ref().unwrap().tensors.iter().position(|t|t.name==name).unwrap();
            (graph,if case==0 {usize::MAX} else {op},if case==1 {usize::MAX} else {tensor})
        });
        let visits=raw_clear_producer::LOOKUP_SPEC_VISITS.with(|n|n.get());
        assert!(raw_clear_producer::reserve_forged_key_for_tests(output,PrimitiveQuote::checked(64,0).unwrap(),
            graph,operation,tensor).is_err());
        // Wrong graph (case 2) must fail BEFORE any concrete lookup_spec call,
        // irrespective of graph sizes. Valid same-graph Linear (case 3) reaches
        // the observer, then fails operation binding; proves observer is live.
        assert_eq!(raw_clear_producer::LOOKUP_SPEC_VISITS.with(|n|n.get())-visits,usize::from(case==3));
        assert!(host.claim_reader().is_err());
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
        host.retire().unwrap(); host.release_final_charges().unwrap();
    }
}

#[test]
fn original_reservation_refuses_equal_content_foreign_root_header_and_backing() {
    for case in 0..3 {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let mut output=genuine_reservation_input(&host);
        let other=Fixture::new().collect(19).unwrap(); other.finish().unwrap();
        match case {
            0=>raw_clear_producer::replace_activation_reader_for_tests(&mut output,other.claim_reader().unwrap()),
            1=>host.inspect(|r|r.header=Fixture::new().header()),
            _=>host.inspect(|r| {
                let fixture=r.header.fixture.as_ref().unwrap();
                let mut writer=PublicFileWriter::new_in(fixture.sizes[2],fixture.hashes[2],
                    Arc::clone(&r.directory) as Arc<dyn SnapshotDirectory>).unwrap();
                writer.write(&fixture.files[2].1).unwrap();
                r.files[2].file=Some(Arc::new(writer.finish().unwrap()));
            }),
        }
        let before=raw_clear_producer::RESERVED_FACTORIES.with(|n|n.get());
        assert!(raw_clear_producer::reserved_test_factory(output,false).is_err());
        assert_eq!(raw_clear_producer::RESERVED_FACTORIES.with(|n|n.get()),before);
        assert!(host.claim_reader().is_err());
    }
}

#[test]
fn original_reservation_demand_is_checked_not_independent_budget_or_authority() {
    assert!(PrimitiveQuote::checked(0,usize::MAX).is_err());
    assert!(PrimitiveQuote::checked(u64::MAX,0).unwrap().include(1,0).is_err());
    assert!(PrimitiveQuote::checked(0,MAX_PROGRAM_BYTES).unwrap().include(0,1).is_err());
    // No extraction/reattachment or second-reservation API exists. Exhaustion is
    // exercised with the real original sibling phase, not a fabricated replay.
}

// Norm SPEC/gamma only. Definitions UNRUN; no norm output/kernel is exercised.
fn genuine_norm_input(host: &RawHost, graph: RawGraph) -> RawLookupActivation {
    let tokens:&[u64]=match graph {RawGraph::Prefill=>&[0,1,2,0],RawGraph::Decode=>&[2]};
    PrivateSourceInstance::new(host.claim_reader().unwrap().admit_tokens(graph,tokens).unwrap()).token_lookup().unwrap()
}
fn norm_counts() -> (usize,usize) {
    (raw_clear_producer::NORM_READS.with(|n|n.get()),raw_clear_producer::NORM_FACTORIES.with(|n|n.get()))
}
fn corrupt_gamma_metadata_after_seal(host: &RawHost, edit: impl FnOnce(&mut ExpectedTensor)) {
    host.inspect(|r| {
        let ops=&r.header.original.plan().prefill.operations;
        let lookup=ops.iter().find(|op|op.operator==pllm_models::ModelOperator::TokenLookup).unwrap();
        let norm=ops.iter().find(|op|op.operator==pllm_models::ModelOperator::RmsNorm && op.inputs==[lookup.id.clone()]).unwrap();
        let name=norm.attributes["weight"].as_str().unwrap();
        let tensor=r.header.raw_premise.as_ref().unwrap().tensors.iter().position(|t|t.name==name).unwrap();
        // Privileged test-only post-registration corruption, NOT a producer API.
        // Restore the SAME Arc identity; no replacement header/source authority.
        r.header.identity=std::sync::Weak::new();
        let header=Arc::get_mut(&mut r.header.header).unwrap();
        edit(&mut header.raw_premise.as_mut().unwrap().tensors[tensor]);
        r.header.identity=Arc::downgrade(&r.header.header);
    });
}

#[test]
fn norm_spec_prefill_decode_original_custody() {
    for graph in [RawGraph::Prefill,RawGraph::Decode] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let norm=genuine_norm_input(&host,graph).admit_input_norm_spec().unwrap();
        let root=Arc::downgrade(&host.root); drop(host);
        let host=RawHost {root:root.upgrade().unwrap()};
        let (shape,epsilon,lookup,operation,tensor,offset,extent,gamma)=norm.snapshot_for_tests();
        assert_eq!(shape,match graph {RawGraph::Prefill=>[1,4,2],RawGraph::Decode=>[1,1,2]});
        assert_eq!(epsilon,0x358637bd); assert_eq!(offset,0); assert_eq!(extent[1]-extent[0],4);
        assert_eq!(gamma,[0x3f80,0x4000]);
        host.inspect(|r| {
            let plan=r.header.original.plan();
            let ops=match graph {RawGraph::Prefill=>&plan.prefill.operations,RawGraph::Decode=>&plan.decode.operations};
            assert_eq!(ops[operation].operator,pllm_models::ModelOperator::RmsNorm);
            assert_eq!(ops[operation].inputs,[ops[lookup].id.clone()]);
            assert_eq!(r.header.raw_premise.as_ref().unwrap().tensors[tensor].name,ops[operation].attributes["weight"].as_str().unwrap());
        });
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
        assert!(host.claim_reader().is_err());
        drop(norm);
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
        host.retire().unwrap(); host.release_final_charges().unwrap(); drop(host);
        assert!(root.upgrade().is_none());
    }
}

#[test]
fn norm_spec_exact_work_one_short_before_reads_factory() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let input=genuine_norm_input(&host,RawGraph::Prefill);
        let quote=raw_clear_producer::norm_quote_for_tests(&input);
        let sibling=leave_primitive_work(&host,quote.work,short);
        let before=norm_counts(); let result=input.admit_input_norm_spec();
        assert_eq!(result.is_err(),short);
        if short { assert_eq!(norm_counts(),before); }
        else { assert_eq!(norm_counts(),(before.0+2,before.1+1)); }
        host.inspect(|r| {
            assert_eq!(sibling.prepay_primitive_work(&r.header.controller,0).is_err(),short);
            if !short { assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err()); }
        });
        drop(result);
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
        host.retire().unwrap(); host.release_final_charges().unwrap();
    }
}

#[test]
fn norm_spec_exact_bytes_one_short_before_reads_factory() {
    for short in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let input=genuine_norm_input(&host,RawGraph::Prefill);
        let quote=raw_clear_producer::norm_quote_for_tests(&input);
        let mut outside=host.inspect(|r|r.header.root.charge_external(0).unwrap());
        let used=outside.budget().used_bytes();
        outside.resize_payload(MAX_PROGRAM_BYTES-used-quote.resident_bytes-size_of::<OwnedExternalCharge>()+usize::from(short)).unwrap();
        let before=norm_counts(); let result=input.admit_input_norm_spec();
        assert_eq!(result.is_err(),short);
        if short { assert_eq!(norm_counts(),before); }
        else { assert_eq!(outside.budget().used_bytes(),MAX_PROGRAM_BYTES); }
        drop(result); drop(outside);
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
        host.retire().unwrap(); host.release_final_charges().unwrap();
    }
}

#[test]
fn norm_spec_exhausted_shared_work() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let input=genuine_norm_input(&host,RawGraph::Prefill);
    let quote=raw_clear_producer::norm_quote_for_tests(&input);
    let sibling=leave_primitive_work(&host,quote.work,false);
    host.inspect(|r|sibling.prepay_primitive_work(&r.header.controller,1).unwrap());
    let before=norm_counts(); assert!(input.admit_input_norm_spec().is_err());
    assert_eq!(norm_counts(),before);
    host.inspect(|r|assert!(sibling.prepay_primitive_work(&r.header.controller,0).is_err()));
    assert!(host.claim_reader().is_err());
}

#[test]
fn norm_spec_epsilon_exact_decimal_component() {
    for epsilon in ["0.000001","1e-6","+1E-0006",".000001","1.e-6","000100e-8","0.0000010000"] {
        assert_eq!(raw_clear_producer::norm_epsilon_for_tests(epsilon).unwrap(),0x358637bd);
    }
    for epsilon in ["","-0.000001","0","-0","1e-5","1e-7","0.0000010000000000000001",
        "1e","1e+","1e--6","1.0.0e-6"," 1e-6","1e-6 ","NaN","inf","1e-99999","1/1000000"] {
        assert!(raw_clear_producer::norm_epsilon_for_tests(epsilon).is_err(),"{epsilon}");
    }
    assert!(raw_clear_producer::norm_epsilon_for_tests(&"0".repeat(NAME+1)).is_err());
    let epsilon=format!("1e-{}6","0".repeat(NAME-5));
    assert_eq!(epsilon.len(),NAME-1);
    assert!(raw_clear_producer::norm_epsilon_for_tests(&epsilon).is_ok());
}

#[test]
fn norm_spec_original_epsilon_equivalents_and_adjacent_rejection() {
    for (epsilon,accepted) in [("+1e-6",true),("0.000001000",true),("1e-5",false),("0.0000010000000000000001",false)] {
        // Independently provision varied ORIGINAL configuration/semantic metadata
        // before candidate ingest, not candidate-derived expected epsilon.
        let mut config:serde_json::Value=serde_json::from_slice(CONFIG).unwrap();
        config["rms_norm_eps"]=serde_json::Value::String(epsilon.into());
        let fixture=Fixture::from_config(&serde_json::to_vec(&config).unwrap());
        let host=fixture.collect(17).unwrap(); host.finish().unwrap();
        let input=genuine_norm_input(&host,RawGraph::Prefill); let before=norm_counts();
        let result=input.admit_input_norm_spec(); assert_eq!(result.is_ok(),accepted);
        if !accepted { assert_eq!(norm_counts(),before); }
        drop(result);
    }
}

#[test]
fn norm_spec_ambiguous_wrongedge_shape_component_only() {
    // OriginalSourcePlan is immutable; this selector-only test cannot mint a
    // capability or pretend a mutated graph is the genuine retained original.
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let (original,lookup)=host.inspect(|r| {
        let ops=r.header.original.plan().prefill.operations.clone();
        let lookup=ops.iter().position(|op|op.operator==pllm_models::ModelOperator::TokenLookup).unwrap(); (ops,lookup)
    });
    let norm=raw_clear_producer::norm_operation_for_tests(&original,lookup,[1,4,2]).unwrap();
    for case in 0..4 {
        let mut ops=original.clone();
        match case {0=>{ops.remove(norm);},1=>{ops.push(ops[norm].clone());},
            2=>ops[norm].inputs=vec!["not.original.lookup".into()],_=>ops[norm].output_shape[2]+=1}
        assert!(raw_clear_producer::norm_operation_for_tests(&ops,lookup,[1,4,2]).is_err());
    }
}

#[test]
fn norm_spec_weight_offset_component_only() {
    assert_eq!(raw_clear_producer::norm_offset_for_tests(Some(&serde_json::json!(0))).unwrap(),0);
    for offset in [serde_json::json!(1),serde_json::json!(-1),serde_json::json!(0.0),serde_json::json!("0"),serde_json::Value::Null] {
        assert!(raw_clear_producer::norm_offset_for_tests(Some(&offset)).is_err());
    }
    assert!(raw_clear_producer::norm_offset_for_tests(None).is_err());
}

#[test]
fn norm_spec_postseal_gamma_metadata_faults() {
    for case in 0..7 {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let input=genuine_norm_input(&host,RawGraph::Prefill);
        corrupt_gamma_metadata_after_seal(&host,|gamma|match case {
            0=>gamma.rank=2,1=>gamma.dtype=raw::RawScalarType::F32Le,2=>gamma.scale=Some("foreign.scale".into()),
            3=>gamma.scale_index=Some(0),4=>gamma.begin+=1,5=>gamma.end+=2,_=>gamma.shape[0]+=1,
        });
        let before=norm_counts(); assert!(input.admit_input_norm_spec().is_err());
        assert_eq!(norm_counts(),before); assert!(host.claim_reader().is_err());
    }
}

#[test]
fn norm_spec_foreign_root_header_equalcontent_backing() {
    for case in 0..3 {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let mut input=genuine_norm_input(&host,RawGraph::Prefill);
        let other=Fixture::new().collect(19).unwrap(); other.finish().unwrap();
        match case {0=>raw_clear_producer::replace_activation_reader_for_tests(&mut input,other.claim_reader().unwrap()),
            1=>host.inspect(|r|r.header=Fixture::new().header()),_=>host.inspect(|r| {
                let fixture=r.header.fixture.as_ref().unwrap();
                let mut writer=PublicFileWriter::new_in(fixture.sizes[2],fixture.hashes[2],Arc::clone(&r.directory) as Arc<dyn SnapshotDirectory>).unwrap();
                writer.write(&fixture.files[2].1).unwrap(); r.files[2].file=Some(Arc::new(writer.finish().unwrap()));
            })}
        let before=norm_counts(); assert!(input.admit_input_norm_spec().is_err()); assert_eq!(norm_counts(),before);
    }
}

#[test]
fn norm_spec_nonfinite_and_late_read_fault_wipe_burn() {
    for nonfinite in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let input=genuine_norm_input(&host,RawGraph::Prefill);
        let quote=raw_clear_producer::norm_quote_for_tests(&input);
        let sibling=leave_primitive_work(&host,quote.work,false);
        let counter=host.inspect(|r|r.header.root.try_clone().unwrap()); let live=counter.budget().used_bytes();
        raw_clear_producer::NORM_GAMMA_DROP.with(|n|n.set(None));
        raw_clear_producer::NORM_SCRATCH_DROP.with(|n|n.set(None));
        // Both faults are injected AFTER genuine registration, lookup and actual
        // original gamma read. Nonfinite second word exercises partial gamma;
        // final-window I/O fault exercises nonzero scratch and empty full capacity.
        raw_clear_producer::NORM_FAIL_READ.with(|n|n.set(if nonfinite {None} else {Some(0)}));
        raw_clear_producer::NORM_NONFINITE_AT.with(|n|n.set(if nonfinite {Some(1)} else {None}));
        let result=input.admit_input_norm_spec();
        raw_clear_producer::NORM_FAIL_READ.with(|n|n.set(None));
        raw_clear_producer::NORM_NONFINITE_AT.with(|n|n.set(None));
        assert!(result.is_err());
        let (wiped,covered)=raw_clear_producer::NORM_GAMMA_DROP.with(|n|n.get()).unwrap();
        let (scratch_wiped,scratch_covered)=raw_clear_producer::NORM_SCRATCH_DROP.with(|n|n.get()).unwrap();
        assert!(wiped && scratch_wiped); assert!(covered>live); assert_eq!(covered,scratch_covered);
        assert!(counter.budget().used_bytes()<live);
        host.inspect(|r| {sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
            assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());});
        assert!(host.claim_reader().is_err());
        assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
        host.retire().unwrap(); host.release_final_charges().unwrap();
    }
}

#[test]
fn norm_spec_multiwindow_second_window_fault_wipes_burns() {
    // Provision geometry BEFORE candidate ingestion; only hidden_size differs.
    // Actual successful construction/ingestion/finish/lookup is REQUIRED below,
    // not a claim that this UNRUN definition passes existing 64MiB/work caps.
    let mut config:serde_json::Value=serde_json::from_slice(CONFIG).unwrap();
    config["hidden_size"]=serde_json::json!(2050);
    let fixture=Fixture::from_config(&serde_json::to_vec(&config).unwrap());
    let host=fixture.collect(FRAME).expect("wide original fixture must ingest under unchanged caps");
    host.finish().expect("wide original fixture must seal under unchanged caps");
    let input=genuine_norm_input(&host,RawGraph::Prefill);
    assert_eq!(FRAME,4096); assert_eq!(input.shape(),[1,4,2050]); assert_eq!(input.words().len(),8200);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
    let quote=raw_clear_producer::norm_quote_for_tests(&input);
    let sibling=leave_primitive_work(&host,quote.work,false);
    let counter=host.inspect(|r|r.header.root.try_clone().unwrap()); let live=counter.budget().used_bytes();
    let before=norm_counts();
    raw_clear_producer::NORM_GAMMA_PREWIPE.with(|n|n.set(None));
    raw_clear_producer::NORM_GAMMA_DROP.with(|n|n.set(None));
    raw_clear_producer::NORM_SCRATCH_DROP.with(|n|n.set(None));
    raw_clear_producer::NORM_READ_FAULT_FIRED.with(|n|n.set(None));
    // AFTER second SUCCESSFUL provider read, before decoding its final 4 bytes.
    // This is a late fault simulation, NOT coverage of a provider I/O error.
    raw_clear_producer::NORM_FAIL_READ.with(|n|n.set(Some(1)));
    let result=input.admit_input_norm_spec();
    raw_clear_producer::NORM_FAIL_READ.with(|n|n.set(None));
    assert!(result.is_err()); assert_eq!(norm_counts(),(before.0+3,before.1+1));
    assert_eq!(raw_clear_producer::NORM_READ_FAULT_FIRED.with(|n|n.get()),Some(1));
    assert_eq!(raw_clear_producer::NORM_GAMMA_PREWIPE.with(|n|n.get()),Some((2048,2050)));
    let (wiped_freed,covered)=raw_clear_producer::NORM_GAMMA_DROP.with(|n|n.get()).unwrap();
    let (scratch_wiped,scratch_covered)=raw_clear_producer::NORM_SCRATCH_DROP.with(|n|n.get()).unwrap();
    assert!(wiped_freed && scratch_wiped); assert!(covered>live); assert_eq!(covered,scratch_covered);
    assert!(counter.budget().used_bytes()<live);
    assert!(host.claim_reader().is_err());
    // Shared meter evidence BEFORE Ready prevents any fresh admission.
    host.inspect(|r| {sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
        assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());});
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    host.retire().unwrap(); host.release_final_charges().unwrap();
}

#[test]
fn norm_spec_byte_one_short_burns_original_work() {
    let fixture=Fixture::new(); let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let input=genuine_norm_input(&host,RawGraph::Prefill);
    let quote=raw_clear_producer::norm_quote_for_tests(&input);
    let sibling=leave_primitive_work(&host,quote.work,false);
    let mut outside=host.inspect(|r|r.header.root.charge_external(0).unwrap());
    let used=outside.budget().used_bytes();
    outside.resize_payload(MAX_PROGRAM_BYTES-used-quote.resident_bytes-size_of::<OwnedExternalCharge>()+1).unwrap();
    let (old_lookup_payload,live)=input.lookup_charge_for_tests();
    assert!(old_lookup_payload>0); assert_eq!(live,outside.budget().used_bytes());
    let before=norm_counts(); let visits=raw_clear_producer::LOOKUP_SPEC_VISITS.with(|n|n.get());
    raw_clear_producer::ORIGINAL_REFUSAL.with(|n|n.set(None));
    raw_clear_producer::NORM_GAMMA_PREWIPE.with(|n|n.set(None));
    raw_clear_producer::NORM_GAMMA_DROP.with(|n|n.set(None));
    raw_clear_producer::NORM_SCRATCH_DROP.with(|n|n.set(None));
    raw_clear_producer::LOOKUP_DROP.with(|n|n.set(None));
    assert!(input.admit_input_norm_spec().is_err());
    assert_eq!(norm_counts(),before);
    assert_eq!(raw_clear_producer::LOOKUP_SPEC_VISITS.with(|n|n.get()),visits);
    // Snapshot AT refusal, before consumed producer cleanup. Actual old charge,
    // not historical resources() or an invented sum, remains fully retained.
    assert_eq!(raw_clear_producer::ORIGINAL_REFUSAL.with(|n|n.get()),Some((8,8,old_lookup_payload,live)));
    assert_eq!(raw_clear_producer::LOOKUP_DROP.with(|n|n.get()),Some((true,live)));
    assert!(outside.budget().used_bytes()<live);
    assert!(raw_clear_producer::NORM_GAMMA_PREWIPE.with(|n|n.get()).is_none());
    assert!(raw_clear_producer::NORM_GAMMA_DROP.with(|n|n.get()).is_none());
    assert!(raw_clear_producer::NORM_SCRATCH_DROP.with(|n|n.get()).is_none());
    host.inspect(|r| {sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
        assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());});
    assert!(host.claim_reader().is_err());
    drop(outside);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    host.retire().unwrap(); host.release_final_charges().unwrap();
    // Quote-derived boundary/debit test, NOT independent quote completeness proof.
}


// Bootstrap repair regressions: source definitions only, UNRUN. All arithmetic
// below is independent of production layout/quote helpers. size_of uses actual
// target layouts, not a hardcoded 64-bit host ABI. Original budget/ledger is
// created ONCE by Fixture::new BEFORE construction; fillers debit that same
// ledger irreversibly, never reset/reissue it or adopt a fresh counter.
fn independent_reader_bytes() -> usize {
    3*size_of::<RawReader>()+2*size_of::<Result<RawReader,Error>>()
        +2*size_of::<std::sync::MutexGuard<'_,RawHostState>>()
        +size_of::<OwnedExternalCharge>()
}
fn independent_tensor_bytes() -> usize {
    3*size_of::<RawTensor<'_>>()+2*size_of::<Result<RawTensor<'_>,Error>>()
}
fn independent_window_bytes() -> usize {
    3*size_of::<RawWindow<'_,'_>>()+2*size_of::<Result<RawWindow<'_,'_>,Error>>()
        +size_of::<std::sync::MutexGuard<'_,RawHostState>>()+8
        +2*size_of::<Arc<PublicFile>>()+4*size_of::<u64>()
        // Actual concrete layouts, independently inventoried multiplicities;
        // no production quote/helper as arithmetic oracle.
        +2*size_of::<LockedReadArgs<'_,'_,'_>>()
        +size_of::<Result<RawWindow<'_,'_>,Error>>()
}
fn independent_primitive_controls() -> usize {
    2*size_of::<std::sync::MutexGuard<'_,RawHostState>>()
        +2*size_of::<OriginalOperationKey>()+3*size_of::<PrimitiveQuote>()
        +2*size_of::<OwnedExternalCharge>()+4*size_of::<Arc<PublicFile>>()
        +2*size_of::<Arc<RootedRawHost>>()+16*size_of::<u64>()
}
// Fully expanded target formulas, never the production quote as prediction.
fn independent_typed_io_controls() -> usize {
    use std::mem::size_of as s;
    use public_file::PublicReadError as E;
    use std::io::Error as I;
    let b=s::<&PublicFile>()+s::<u64>()+s::<&mut [u8]>();
    let k=s::<&std::fs::File>()+s::<&mut [u8]>()+s::<u64>();
    #[cfg(unix)]
    let platform=s::<Result<(),I>>();
    #[cfg(windows)]
    let platform=s::<Result<usize,I>>()+5*s::<usize>()+s::<std::ops::RangeFrom<usize>>()
        +2*s::<u64>()+2*s::<bool>();
    2*b+3*k+s::<usize>()+s::<Option<u64>>()+4*s::<u64>()+4*s::<bool>()
        +3*s::<Result<(),E>>()+s::<E>()+s::<I>()+platform
}
fn independent_raw_shared_io_controls() -> usize {
    use std::mem::size_of as s;
    use public_file::PublicReadError as E;
    use std::io::Error as I;
    // Extra legacy method call transfer/parameters, not typed method's own 2B.
    let b=s::<&PublicFile>()+s::<u64>()+s::<&mut [u8]>();
    independent_typed_io_controls()+2*b+6*s::<E>()+3*s::<I>()+s::<&I>()
        +8*s::<String>()+2*s::<Result<(),String>>()
}
fn independent_registration_reader_io_bootstrap_bytes() -> usize {
    // Entry lock/result + guards; all authority is borrowed, never reconstructed.
    let entry=size_of::<std::sync::LockResult<std::sync::MutexGuard<'_,RawHostState>>>()
        +size_of::<Result<std::sync::MutexGuard<'_,RawHostState>,Error>>()
        +2*size_of::<std::sync::MutexGuard<'_,RawHostState>>()
        +size_of::<&RawHost>()+size_of::<&RawReader>()+size_of::<&RawTensor<'_>>()
        +2*size_of::<&RawRegistration>()+size_of::<Result<&RawRegistration,Error>>()
        +size_of::<&RawPending>()+size_of::<&SourceHeaderEntry>()+size_of::<&OriginalSourcePlan>()
        +size_of::<&OwnedExternalCharge>();
    let quotation=size_of::<RawGraph>()+size_of::<OriginalOperationKey>()
        +size_of::<&OriginalOperationKey>()+size_of::<&pllm_models::DecoderPlan>()
        +size_of::<&TrustedRawPremise>()+size_of::<Result<&TrustedRawPremise,Error>>()
        +3*size_of::<PrimitiveQuote>()+2*size_of::<Result<PrimitiveQuote,Error>>()
        +3*size_of::<usize>()+4*size_of::<usize>()
        +size_of::<std::array::IntoIter<usize,4>>();
    let arithmetic=16*(size_of::<usize>()+size_of::<Option<usize>>()
        +size_of::<Result<usize,Error>>())
        +4*(size_of::<u64>()+size_of::<Option<u64>>()+size_of::<Result<u64,Error>>())
        +size_of::<Result<u64,std::num::TryFromIntError>>();
    let admission=2*size_of::<Result<(),Error>>()
        +2*(size_of::<OwnedExternalCharge>()+size_of::<Result<OwnedExternalCharge,Error>>());
    let returns=size_of::<Result<RawReader,Error>>()
        +size_of::<Result<RawTensor<'_>,Error>>()
        // Helper Err and wrapper Err outlive the released demand charge.
        +2*size_of::<Result<RawWindow<'_,'_>,Error>>();
    entry+quotation+arithmetic+admission+returns+independent_raw_shared_io_controls()
}
fn independent_arc_bytes<T>() -> usize {
    let (layout,_)=std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize;2]>()
        .extend(std::alloc::Layout::new::<T>()).unwrap();
    layout.pad_to_align().size()
}
fn independent_phase_bytes(h: &SourceHeaderEntry, f: usize, t: usize) -> usize {
    let ops=h.original.plan().prefill.operations.len();
    let metadata=f*(size_of::<ExpectedFile>()+NAME)+t*(size_of::<ExpectedTensor>()+2*NAME);
    let shapes=ops*(source_registry::tree_entry_bytes::<(String,Vec<usize>)>().unwrap()
        +source_registry::tree_entry_bytes::<(String,usize)>().unwrap()+2*NAME+2*size_of::<usize>());
    let initial=h.collection_charge.payload_bytes()+independent_registration_reader_io_bootstrap_bytes()
        +2*source_policy::CONFIG_BYTES+2*metadata+2*shapes+3*size_of::<TrustedRawPremise>()
        +size_of::<OwnedBudget>()+3*size_of::<HostRawExpectations<'_>>()
        +canonical::resources::<std::io::Sink>(16).unwrap().hash_workspace_bytes;
    let host_controls=2*(size_of::<RawRegistration>()+size_of::<RawHostState>()
        +size_of::<RawHost>()+size_of::<RawPending>()
        +size_of::<std::sync::MutexGuard<'_,RawHostState>>()
        +size_of::<Result<RetirementReadiness,Error>>())
        +independent_arc_bytes::<RootedRawHost>()+independent_arc_bytes::<SourceHeaderEntry>();
    let phase=initial+f*(size_of::<RawBacking>()+independent_arc_bytes::<PublicFile>()
        +public_file::backing_bytes().unwrap())+public_file::writer_creation_bytes().unwrap()
        +host_controls+public_file::MAX_CHUNK_BYTES+2*JSON_BYTES+json::bootstrap_bytes().unwrap()
        +2*FRAME+size_of::<SourceDecodeWorkspace>()+4*size_of::<Sha256>();
    #[cfg(feature="lifecycle-test-support")]
    let phase={
        use crate::replicated_decoder::source::Publisher;
        phase+3*std::alloc::Layout::new::<Publisher>().pad_to_align().size()
            +2*std::alloc::Layout::new::<Option<Publisher>>().pad_to_align().size()
            +2*size_of::<Result<Publisher,String>>()+4*size_of::<[usize;10]>()
            +2*size_of::<Result<[usize;10],Error>>()
            +3*(size_of::<(String,Vec<u8>)>()+NAME+JSON_BYTES)
            +3*size_of::<Vec<(String,Vec<u8>)>>()+3*size_of::<[u8;17]>()
            +2*size_of::<Result<([u8;17],usize),Error>>()
            +3*size_of::<(u64,[u8;32],usize)>()+2*size_of::<Result<(u64,[u8;32],usize),Error>>()
    };
    phase
}
fn repair_probes() -> [usize;7] { READER_PROBES.with(|n|n.get()) }
fn reset_repair_probes() { READER_PROBES.with(|n|n.set([0;7])); }
fn independent_primitive_work(host: &RawHost, demand: u64) -> u64 {
    host.inspect(|r| {
        let o=r.header.original.plan().prefill.operations.len() as u64;
        let t=r.header.raw_premise.as_ref().unwrap().tensors.len() as u64;
        let f=r.files.len() as u64;
        demand+2*(o*(4*1024+64)+t*(1024+1)+64*f+1024)
            +independent_primitive_controls() as u64
    })
}
fn repair_key(host: &RawHost) -> OriginalOperationKey {
    host.inspect(|r| {
        let operation=r.header.original.plan().prefill.operations.iter()
            .position(|op|op.operator==pllm_models::ModelOperator::TokenLookup).unwrap();
        let name=r.header.original.plan().prefill.operations[operation].attributes["weight"].as_str().unwrap();
        let tensor=r.header.raw_premise.as_ref().unwrap().tensors.iter().position(|t|t.name==name).unwrap();
        OriginalOperationKey {graph:RawGraph::Prefill,operation,tensor}
    })
}

#[test]
fn raw_shared_typed_io_controls_are_initial_only_without_demand_surcharge() {
    assert_eq!(public_file::READ_UNFORMATTED_CONTROL_BYTES,independent_typed_io_controls());
    assert_eq!(RAW_SHARED_IO_CONTROL_BYTES,independent_raw_shared_io_controls());
    assert_eq!(REGISTRATION_READER_IO_BOOTSTRAP_BYTES,independent_registration_reader_io_bootstrap_bytes());
    // No shared I/O surcharge in claim/tensor/primitive/window demands. Their
    // original independently reconstructed exact/one-short tests remain intact.
    assert_eq!(reader_control_bytes().unwrap(),independent_reader_bytes());
    assert_eq!(TENSOR_CONTROL_BYTES,independent_tensor_bytes());
    assert_eq!(PRIMITIVE_CONTROL_BYTES,independent_primitive_controls());
    assert_eq!(raw_window_bytes().unwrap(),independent_window_bytes());
    assert_eq!(RAW_WINDOW_WORK,(8*4096+1024+256+4) as u64);
    assert_eq!((PRIMITIVE_BOOTSTRAP_WORK,WEIGHT_BOOTSTRAP_WORK,CLAIM_BOOTSTRAP_WORK,READ_BOOTSTRAP_WORK),(24,7,1,1));
    assert_eq!(READER_CLAIM_WORK,64);
}

#[test]
fn raw_bootstrap_const_layouts_and_retained_initial_inventory_are_independent() {
    assert_eq!(reader_control_bytes().unwrap(),independent_reader_bytes());
    assert_eq!(TENSOR_CONTROL_BYTES,independent_tensor_bytes());
    assert_eq!(raw_window_bytes().unwrap(),independent_window_bytes());
    assert_eq!(PRIMITIVE_CONTROL_BYTES,independent_primitive_controls());
    assert_eq!(PRIMITIVE_CONTROL_WORK,independent_primitive_controls() as u64);
    assert_eq!(REGISTRATION_READER_IO_BOOTSTRAP_BYTES,independent_registration_reader_io_bootstrap_bytes());
    let mut owned=Fixture::new().into_collector().unwrap();
    let SourceCollector {payload,header}=&mut owned;
    let fixture=payload.as_ref().unwrap();
    let expected=independent_phase_bytes(header,fixture.files.len(),fixture.tensors.len());
    let premise=fixture.with_expected(|e|TrustedRawPremise::provision(header,e)).unwrap().unwrap();
    assert_eq!(premise.phase_bytes,expected);
    assert_eq!(header.collection_charge.payload_bytes(),expected);
    // This original charge survives freeze, reader failure and reader Drop;
    // existing Busy/custody/wipe regressions cover its eventual release ordering.
}

#[test]
fn raw_bootstrap_four_boundaries_distinguish_quote_from_original_demand() {
    // Paths: claim, weight, primitive, read. The constants here are independently
    // approved event prices, not values fetched from the implementation.
    for path in 0..4 {
        for boundary in 0..4 {
            let fixture=Fixture::new();
            let seed=fixture.seed.as_ref().unwrap();
            let counter=seed.root.try_clone().unwrap(); // identity before construction
            let host=fixture.collect(17).unwrap(); host.finish().unwrap();
            host.inspect(|r|assert!(std::ptr::eq(counter.budget(),r.header.original.budget())));
            let key=repair_key(&host);
            let reader=if path==0 {None} else {Some(host.claim_reader().unwrap())};
            let tensor=if path==3 {Some(reader.as_ref().unwrap().weight(RawGraph::Prefill,key.operation).unwrap())} else {None};
            let b=[1,7,24,1][path];
            let w=match path {
                0=>64,
                1=>host.inspect(|r|r.header.raw_premise.as_ref().unwrap().tensors.len() as u64*1025+128),
                2=>independent_primitive_work(&host,19),
                _=>8*4096+1024+256+4,
            };
            let remaining=[b-1,b,b+w-1,b+w][boundary];
            // U is the SAME original cumulative consumption after construction,
            // claim/tensor preparation and sibling lowering. No ledger reset.
            let sibling=leave_reader_work(&host,remaining,false);
            reset_repair_probes();
            let factories=(READER_FACTORIES.with(|n|n.get()),WINDOW_FACTORIES.with(|n|n.get()));
            let allocation_trap=RepairAllocationTrap::new();
            let success=match path {
                0=>host.claim_reader().is_ok(),
                1=>reader.as_ref().unwrap().weight(RawGraph::Prefill,key.operation).is_ok(),
                2=>admit_primitive(reader.as_ref().unwrap(),&key,PrimitiveQuote::checked(19,0).unwrap()).is_ok(),
                _=>tensor.as_ref().unwrap().read(0,2).is_ok(),
            };
            assert_eq!(success,boundary==3);
            drop(allocation_trap);
            let probes=repair_probes();
            assert_eq!(probes[5],0);
            if boundary==0 { assert_eq!(probes,[0;7]); }
            else {
                assert_eq!(probes[0],usize::from(path==2));
                assert_eq!(probes[1],usize::from(path==1));
                if boundary<3 { assert_eq!(&probes[2..],&[0;5]); }
            }
            if boundary<3 {
                assert_eq!(READER_FACTORIES.with(|n|n.get()),factories.0);
                assert_eq!(WINDOW_FACTORIES.with(|n|n.get()),factories.1);
                // Preexisting failure/claim cannot be reset or charged again.
                let before=repair_probes();
                if path==0 { assert!(host.claim_reader().is_err()); }
                else { assert!(reader.as_ref().unwrap().weight(RawGraph::Prefill,key.operation).is_err()); }
                assert_eq!(repair_probes(),before);
            }
            host.inspect(|r| {
                // Bj-1 burns at bootstrap; Bj and Bj+Wj-1 burn at demand.
                // Exact U+Bj+Wj succeeds and remains fully spent after Drop.
                assert_eq!(sibling.prepay_primitive_work(&r.header.controller,0).is_err(),boundary<3);
                if boundary==3 { assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err()); }
            });
            drop(tensor); drop(reader);
        }
    }
}

#[test]
fn raw_bootstrap_second_success_is_cumulative_then_exhaustion_is_terminal() {
    for primitive in [false,true] {
        let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
        let reader=host.claim_reader().unwrap(); let key=repair_key(&host);
        let work=if primitive {24+independent_primitive_work(&host,11)} else {1+8*4096+1024+256+4};
        let tensor=if primitive {None} else {Some(reader.weight(RawGraph::Prefill,key.operation).unwrap())};
        let sibling=leave_reader_work(&host,2*work,false);
        for _ in 0..2 {
            if primitive { drop(admit_primitive(&reader,&key,PrimitiveQuote::checked(11,0).unwrap()).unwrap()); }
            else { drop(tensor.as_ref().unwrap().read(0,2).unwrap()); }
        }
        // Zero query proves unburned at exact cumulative U+2*(B+W), after Drop.
        host.inspect(|r|sibling.prepay_primitive_work(&r.header.controller,0).unwrap());
        if primitive { assert!(admit_primitive(&reader,&key,PrimitiveQuote::checked(11,0).unwrap()).is_err()); }
        else { assert!(tensor.as_ref().unwrap().read(0,2).is_err()); }
        assert!(host.root.state.lock().unwrap().reader_failed);
        reset_repair_probes();
        assert!(reader.weight(RawGraph::Prefill,key.operation).is_err());
        assert_eq!(repair_probes(),[0;7]);
        host.inspect(|r|assert!(sibling.prepay_primitive_work(&r.header.controller,0).is_err()));
    }
}

#[test]
fn raw_bootstrap_late_io_callsite_faults_keep_debit_latch_and_wipe_custody() {
    for primitive in [false,true] {
        for fault in 0..if primitive {1} else {2} {
            let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
            let reader=host.claim_reader().unwrap(); let key=repair_key(&host);
            let tensor=if primitive {None} else {Some(reader.weight(RawGraph::Prefill,key.operation).unwrap())};
            let work=if primitive {24+independent_primitive_work(&host,17)} else {1+8*4096+1024+256+4};
            let sibling=leave_reader_work(&host,work,false);
            let (cover,used)=host.inspect(|r|(r.header.collection_charge.payload_bytes(),r.header.root.budget().used_bytes()));
            reset_repair_probes(); WINDOW_DROP.with(|n|n.set(None));
            // Test-only faults AT actual I/O call sites, not success publication.
            READER_IO_FAULT.with(|n|n.set(Some(fault)));
            if primitive { assert!(admit_primitive(&reader,&key,PrimitiveQuote::checked(17,0).unwrap()).is_err()); }
            else {
                let result=tensor.as_ref().unwrap().read(0,2);
                assert!(result.is_err());
                // Actual helper -> wrapper error return: demand charge is gone,
                // yet full result forwarding remains covered by the ORIGINAL
                // retained bootstrap frame (two R slots), not released demand.
                host.inspect(|r| {
                    assert_eq!(r.header.root.budget().used_bytes(),used);
                    assert_eq!(r.header.collection_charge.payload_bytes(),cover);
                    assert!(cover>=independent_registration_reader_io_bootstrap_bytes());
                });
                drop(result);
                host.inspect(|r| {
                    assert_eq!(r.header.root.budget().used_bytes(),used);
                    assert_eq!(r.header.collection_charge.payload_bytes(),cover);
                });
            }
            READER_IO_FAULT.with(|n|n.set(None));
            assert_eq!(repair_probes()[3],1);
            assert_eq!(repair_probes()[4],usize::from(fault==1));
            if fault==1 {
                let (wiped,covered)=WINDOW_DROP.with(|n|n.get()).unwrap();
                assert!(wiped); assert_eq!(covered,used+independent_window_bytes()+size_of::<OwnedExternalCharge>());
            } else { assert!(WINDOW_DROP.with(|n|n.get()).is_none()); }
            assert!(host.root.state.lock().unwrap().reader_failed);
            assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
            host.inspect(|r| {
                assert_eq!(r.header.collection_charge.payload_bytes(),cover);
                assert_eq!(r.header.root.budget().used_bytes(),used);
                sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
            });
            reset_repair_probes(); assert!(reader.weight(RawGraph::Prefill,key.operation).is_err());
            assert_eq!(repair_probes(),[0;7]);
            drop(tensor); drop(reader);
            assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
            host.retire().unwrap();
            host.inspect(|r| {
                assert_eq!(r.header.collection_charge.payload_bytes(),cover);
                assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());
            });
            host.release_final_charges().unwrap();
        }
    }
}

#[test]
fn raw_bootstrap_quote_failure_is_reader_local_and_does_not_abandon_source() {
    let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
    let reader=host.claim_reader().unwrap(); let key=repair_key(&host);
    let sibling=leave_reader_work(&host,24+5,false);
    reset_repair_probes();
    assert!(admit_primitive(&reader,&key,PrimitiveQuote::checked(u64::MAX,0).unwrap()).is_err());
    assert_eq!(repair_probes(),[1,0,0,0,0,0,0]);
    assert!(host.root.state.lock().unwrap().reader_failed);
    assert!(reader.weight(RawGraph::Prefill,key.operation).is_err());
    // Only bootstrap spent: full demand never admitted, source not abandoned.
    host.inspect(|r| {
        sibling.prepay_primitive_work(&r.header.controller,5).unwrap();
        sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
    });
    drop(reader);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
}


#[test]
fn raw_bootstrap_independent_exact_storage_all_paths_and_old_window_peak() {
    for path in 0..4 {
        for short in [false,true] {
            let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
            let key=repair_key(&host);
            let reader=if path==0 {None} else {Some(host.claim_reader().unwrap())};
            let tensor=if path==3 {Some(reader.as_ref().unwrap().weight(RawGraph::Prefill,key.operation).unwrap())} else {None};
            let old=if path==3 {Some(tensor.as_ref().unwrap().read(0,2).unwrap())} else {None};
            let work=match path {
                0=>1+64,
                1=>7+host.inspect(|r|r.header.raw_premise.as_ref().unwrap().tensors.len() as u64*1025+128),
                2=>24+independent_primitive_work(&host,23),
                _=>1+8*4096+1024+256+4,
            };
            let sibling=leave_reader_work(&host,work,false);
            let bytes=match path {
                0=>independent_reader_bytes(),1=>independent_tensor_bytes(),
                2=>independent_primitive_controls()+37,_=>independent_window_bytes(),
            }+size_of::<OwnedExternalCharge>();
            let mut outside=host.inspect(|r|r.header.root.charge_external(0).unwrap());
            let before=outside.budget().used_bytes();
            outside.resize_payload(MAX_PROGRAM_BYTES-before-bytes+usize::from(short)).unwrap();
            let baseline=outside.budget().used_bytes();
            let counts=norm_counts(); reset_repair_probes();
            // Each result remains alive while checking the independently
            // reconstructed actual counter peak, not merely a quote return.
            macro_rules! check_peak {
                ($result:expr)=>{{
                    let result=$result;
                    assert_eq!(result.is_err(),short);
                    assert_eq!(outside.budget().used_bytes(),if short {baseline} else {MAX_PROGRAM_BYTES});
                    drop(result);
                }};
            }
            match path {
                0=>check_peak!(host.claim_reader()),
                1=>check_peak!(reader.as_ref().unwrap().weight(RawGraph::Prefill,key.operation)),
                2=>check_peak!(admit_primitive(reader.as_ref().unwrap(),&key,PrimitiveQuote::checked(23,37).unwrap())),
                _=>check_peak!(tensor.as_ref().unwrap().read(2,2)),
            }
            // Reader controls deliberately remain in reader_charge after
            // reader Drop; tensor/window/reservation charges are per-call.
            assert_eq!(outside.budget().used_bytes(),
                if path==0 && !short {MAX_PROGRAM_BYTES} else {baseline});
            if short {
                assert_eq!(norm_counts(),counts);
                assert_eq!(&repair_probes()[2..],&[0;5]);
            }
            if let Some(old)=old.as_ref() { assert_eq!(old.bytes(),[0x80,0x3f]); }
            host.inspect(|r| {
                // Storage refusal does not refund the successful B+W debits,
                // nor does it burn the source as an abandoned PrimitiveAttempt.
                sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
                assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());
            });
            drop(outside); drop(old); drop(tensor); drop(reader);
        }
    }
}

#[test]
fn raw_bootstrap_weight_quote_failure_spends_only_bootstrap_and_latches() {
    let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
    let reader=host.claim_reader().unwrap(); let key=repair_key(&host);
    let sibling=leave_reader_work(&host,7+3,false);
    {
        // Privileged test corruption of retained premise; no production setter,
        // alternate authority, budget replacement or callback is introduced.
        let mut state=host.root.state.lock().unwrap(); let r=state.payload.as_mut().unwrap();
        r.header.identity=std::sync::Weak::new();
        Arc::get_mut(&mut r.header.header).unwrap().raw_premise.take();
        r.header.identity=Arc::downgrade(&r.header.header);
    }
    reset_repair_probes();
    assert!(reader.weight(RawGraph::Prefill,key.operation).is_err());
    assert_eq!(repair_probes(),[0,1,0,0,0,0,0]);
    assert!(host.root.state.lock().unwrap().reader_failed);
    assert!(reader.weight(RawGraph::Prefill,key.operation).is_err());
    assert_eq!(repair_probes(),[0,1,0,0,0,0,0]);
    drop(reader);
    host.inspect(|r| {
        sibling.prepay_primitive_work(&r.header.controller,3).unwrap();
        sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
        assert!(sibling.prepay_primitive_work(&r.header.controller,1).is_err());
    });
}

#[test]
fn raw_bootstrap_controller_precondition_failure_is_not_a_ledger_burn() {
    let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
    let reader=host.claim_reader().unwrap(); let key=repair_key(&host);
    let sibling=leave_reader_work(&host,9,false);
    let capacity={
        let mut state=host.root.state.lock().unwrap(); let r=state.payload.as_mut().unwrap();
        r.header.identity=std::sync::Weak::new();
        let h=Arc::get_mut(&mut r.header.header).unwrap();
        let capacity=h.controller.payload_bytes(); h.controller.resize_payload(0).unwrap();
        r.header.identity=Arc::downgrade(&r.header.header); capacity
    };
    reset_repair_probes();
    assert!(reader.weight(RawGraph::Prefill,key.operation).is_err());
    assert_eq!(repair_probes(),[0;7]);
    assert!(host.root.state.lock().unwrap().reader_failed);
    {
        let mut state=host.root.state.lock().unwrap(); let r=state.payload.as_mut().unwrap();
        r.header.identity=std::sync::Weak::new();
        Arc::get_mut(&mut r.header.header).unwrap().controller.resize_payload(capacity).unwrap();
        r.header.identity=Arc::downgrade(&r.header.header);
    }
    // Restoring only test-corrupted storage cannot reset the failed reader.
    assert!(reader.weight(RawGraph::Prefill,key.operation).is_err());
    assert_eq!(repair_probes(),[0;7]);
    host.inspect(|r| {
        sibling.prepay_primitive_work(&r.header.controller,9).unwrap();
        sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
    });
}


// Test-only allocation factory trap; restore even during assertion unwind.
struct RepairAllocationTrap;
impl RepairAllocationTrap {
    fn new() -> Self { READER_ALLOC_FAULT.with(|n|n.set(true)); Self }
}
impl Drop for RepairAllocationTrap {
    fn drop(&mut self) { READER_ALLOC_FAULT.with(|n|n.set(false)); }
}

#[test]
fn raw_bootstrap_weight_second_success_then_cumulative_exhaustion() {
    let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
    let reader=host.claim_reader().unwrap(); let key=repair_key(&host);
    let work=7+host.inspect(|r|r.header.raw_premise.as_ref().unwrap().tensors.len() as u64*1025+128);
    let sibling=leave_reader_work(&host,2*work,false);
    for _ in 0..2 { drop(reader.weight(RawGraph::Prefill,key.operation).unwrap()); }
    host.inspect(|r|sibling.prepay_primitive_work(&r.header.controller,0).unwrap());
    reset_repair_probes();
    assert!(reader.weight(RawGraph::Prefill,key.operation).is_err());
    assert_eq!(repair_probes(),[0;7]); // bootstrap exhausted before quotation
    assert!(host.root.state.lock().unwrap().reader_failed);
    assert!(reader.weight(RawGraph::Prefill,key.operation).is_err());
    assert_eq!(repair_probes(),[0;7]);
    drop(reader);
    host.inspect(|r|assert!(sibling.prepay_primitive_work(&r.header.controller,0).is_err()));
}

#[test]
fn raw_bootstrap_poisoned_host_lock_never_debits_or_quotes() {
    let host=Fixture::new().collect(17).unwrap(); host.finish().unwrap();
    let reader=host.claim_reader().unwrap(); let key=repair_key(&host);
    let sibling=leave_reader_work(&host,13,false);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard=host.root.state.lock().unwrap();
        panic!("test-only poisoned raw host lock");
    })).is_err());
    reset_repair_probes();
    assert!(reader.weight(RawGraph::Prefill,key.operation).is_err());
    assert!(admit_primitive(&reader,&key,PrimitiveQuote::checked(1,0).unwrap()).is_err());
    assert!(host.claim_reader().is_err());
    assert_eq!(repair_probes(),[0;7]);
    let state=match host.root.state.lock() {Err(error)=>error.into_inner(),Ok(_)=>panic!("lock not poisoned")};
    let r=state.payload.as_ref().unwrap();
    sibling.prepay_primitive_work(&r.header.controller,13).unwrap();
    sibling.prepay_primitive_work(&r.header.controller,0).unwrap();
}

#[test]
fn raw_bootstrap_registration_cover_survives_freeze_busy_retirement_and_reader_drop() {
    let fixture=Fixture::new();
    let counter=fixture.seed.as_ref().unwrap().root.try_clone().unwrap();
    let host=fixture.collect(17).unwrap(); host.finish().unwrap();
    let cover=host.inspect(|r|r.header.collection_charge.payload_bytes());
    assert!(cover>=independent_registration_reader_io_bootstrap_bytes());
    let reader=host.claim_reader().unwrap(); let key=repair_key(&host);
    let tensor=reader.weight(RawGraph::Prefill,key.operation).unwrap();
    let window=tensor.read(0,2).unwrap();
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
    host.inspect(|r|assert_eq!(r.header.collection_charge.payload_bytes(),cover));
    let root=Arc::downgrade(&host.root); drop(host);
    let host=RawHost {root:root.upgrade().unwrap()};
    host.inspect(|r|assert_eq!(r.header.collection_charge.payload_bytes(),cover));
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Busy);
    drop(window); drop(tensor); drop(reader);
    assert_eq!(host.begin_retire().unwrap(),RetirementReadiness::Ready);
    host.inspect(|r|assert_eq!(r.header.collection_charge.payload_bytes(),cover));
    host.retire().unwrap();
    let covered=counter.budget().used_bytes();
    host.release_final_charges().unwrap();
    assert!(counter.budget().used_bytes()<covered);
    drop(host); assert!(root.upgrade().is_none());
}
