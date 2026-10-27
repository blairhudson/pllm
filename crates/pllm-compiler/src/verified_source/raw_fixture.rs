//! Synthetic D31 fixture, not checkpoint authentication or trained-model evidence.
//! Expected native inventory is frozen before any candidate ingestion. Python
//! can read candidate chunks, never replace expectations or repin their hashes.
use super::*;
use std::io::{Cursor, Write as _};

const SHARD: &str = "synthetic-00001.safetensors";
type TensorView<'a> = (&'a str, usize, &'a str, &'a [usize], u64, u64, Option<&'a str>);

#[cfg(test)]
thread_local! {
    // Fixture, Tensor, body, semantic/file digest FACTORY events, not CPU work.
    pub(super) static GENERATION_EVENTS: std::cell::Cell<[usize;4]> = const { std::cell::Cell::new([0;4]) };
    static METADATA_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static GENERATION_DROP: std::cell::Cell<Option<(bool,usize,usize)>> = const { std::cell::Cell::new(None) };
}
fn generation_event(_kind: usize) {
    #[cfg(test)] GENERATION_EVENTS.with(|n| {let mut v=n.get(); v[_kind]+=1; n.set(v);});
}

/// Logical owned-byte and named-pass bounds, not instructions/crypto CPU/RSS.
/// Counts include repeated source names (therefore bound the unique shape map).
/// No maps, Strings, descriptors, body bytes or digests are built by this quote.
#[derive(Clone, Copy)]
pub(super) struct GenerationQuote {
    pub(super) descriptors:usize, pub(super) body:usize, pub(super) header:usize,
    pub(super) index:usize, pub(super) payload:usize, pub(super) resident:usize,
    pub(super) work:u64,
}
impl GenerationQuote {
    // Allocation-free fixed slot inventory. Reserve the complete existing
    // control group early, including three old/new quotation slots; some
    // producer controls are only used later. No header allowance is borrowed.
    // The full resident quote includes this SAME group exactly once.
    fn fixed_control_bytes() -> usize {
        3*size_of::<SourceCollector<Fixture>>()+3*size_of::<RawPending>()
            +3*size_of::<HostRawExpectations<'_>>()+4*size_of::<Sha256>()
            +size_of::<Cursor<&mut [u8]>>()+size_of::<serde_json::Serializer<Cursor<&mut [u8]>>>()
            +size_of::<std::collections::BTreeMap<String,Vec<usize>>>()
            +size_of::<std::collections::BTreeMap<String,usize>>()+3*size_of::<GenerationQuote>()
            +size_of::<[(&str,Vec<u8>);3]>()+3*size_of::<[u8;32]>()
    }
    pub(super) fn census_work(seed:&SourceHeaderEntry) -> Result<u64,Error> {
        let n=seed.original.plan().prefill.operations.len();
        if n>mul(source_policy::DESCRIPTORS,2)? {return Err(Error::Bounds);}
        // Two attributes/op, preceding-input search, identifier comparisons,
        // dimensions/count arithmetic. The canonical allowance covers the
        // retained plan's bounded attribute traversal; no hashing occurs here.
        u64::try_from(plus(CANONICAL_WORK,mul(mul(4,plus(n,1)?)?,mul(plus(n,1)?,NAME)?)?)?).map_err(|_|Error::Bounds)
    }
    pub(super) fn admit(seed:&mut SourceHeaderEntry) -> Result<Self,Error> {
        let existing=seed.collection_charge.payload_bytes();
        // Shared admission checks bytes before work. Debit census first on the
        // retained reservation, then reserve its fixed controls before metadata
        // entry. A short byte reservation must not refund successful work.
        seed.admit_collection(0,existing,Self::census_work(seed)?)?;
        seed.admit_collection(0,plus(existing,Self::fixed_control_bytes())?,0)?;
        let q=Self::metadata(seed)?;
        // Preserve the complete original debit even if subsequent byte growth
        // refuses. Existing admit_collection checks bytes before work, so prepay
        // against the retained capacity first; no new ledger/attempt is issued.
        seed.admit_collection(0,seed.collection_charge.payload_bytes(),q.work)?;
        // q.resident already includes the bootstrap. Add to the ORIGINAL base,
        // not the now-grown charge: preserve excess without counting it twice.
        seed.admit_collection(0,plus(existing,q.resident)?,0)?;
        Ok(q)
    }
    // Caller must prepay census_work AND fixed_control_bytes on THIS seed
    // before entering this pass or constructing its result slots.
    pub(super) fn metadata(seed:&SourceHeaderEntry) -> Result<Self,Error> {
        #[cfg(test)] METADATA_VISITS.with(|n|n.set(n.get()+1));
        use pllm_models::ModelOperator as Op;
        let ops=&seed.original.plan().prefill.operations;
        let vocab=ops.iter().find(|o|o.operator==Op::OutputHead)
            .and_then(|o|o.output_shape.get(1)).copied().ok_or(Error::Binding)?;
        let vocab=usize::try_from(vocab).map_err(|_|Error::Bounds)?;
        let (mut descriptors,mut names,mut words)=(1usize,"zz_unused.weight".len(),6usize);
        for (i,op) in ops.iter().enumerate() {
            if op.id.len()>NAME {return Err(Error::Bounds);}
            let out=usize::try_from(*op.output_shape.last().ok_or(Error::Binding)?).map_err(|_|Error::Bounds)?;
            for key in ["weight","bias"] {
                let Some(name)=op.attributes.get(key).and_then(|v|v.as_str()) else {continue};
                if name.len()>NAME || name=="zz_unused.weight" {return Err(Error::Binding);}
                let count=if key=="bias" {out} else {match op.operator {
                    Op::TokenLookup=>mul(vocab,out)?, Op::RmsNorm=>out,
                    Op::Linear|Op::OutputHead=> {
                        let input=op.inputs.first().ok_or(Error::Binding)?;
                        if input.len()>NAME {return Err(Error::Bounds);}
                        let source=ops[..i].iter().rev().find(|o|&o.id==input).ok_or(Error::Binding)?;
                        mul(out,usize::try_from(*source.output_shape.last().ok_or(Error::Binding)?).map_err(|_|Error::Bounds)?)?
                    }, _=>return Err(Error::Binding),
                }};
                descriptors=plus(descriptors,1)?; names=plus(names,name.len())?; words=plus(words,count)?;
            }
        }
        if descriptors>source_policy::DESCRIPTORS {return Err(Error::Bounds);}
        let body=mul(words,2)?;
        // Six bytes per UTF-8 source byte bounds JSON escaping; rank <=2,
        // usize/u64 integers <=20 decimal digits. Punctuation includes a comma
        // per entry (also the final entry), so empty/singleton cases are covered.
        let record=r#""":{"data_offsets":[,],"dtype":"BF16","shape":[,]},"#.len();
        let header=plus(2,plus(mul(6,names)?,mul(descriptors,plus(record,4*20)?)?)?)?;
        let index=plus(r#"{"weight_map":{}}"#.len(),plus(mul(6,names)?,mul(descriptors,SHARD.len()+6)?)?)?;
        let payload=plus(8,plus(header,body)?)?;
        if payload as u64>public_file::MAX_BYTES {return Err(Error::Bounds);}
        let config=seed.original.raw_configuration().len();
        let map_slots=plus(mul(descriptors,plus(source_registry::tree_entry_bytes::<(String,Vec<usize>)>()?,2*size_of::<usize>())?)?,names)?;
        let known=mul(ops.len(),plus(source_registry::tree_entry_bytes::<(String,usize)>()?,NAME)?)?;
        let tensor_slots=plus(mul(descriptors,size_of::<Tensor>()+2*size_of::<usize>()+4)?,names)?;
        let views=plus(3*size_of::<(&str,u64,[u8;32])>(),mul(descriptors,size_of::<TensorView<'_>>())?)?;
        let files=3*(size_of::<(String,Vec<u8>)>()+size_of::<u64>()+size_of::<[u8;32]>())
            +"config.json".len()+"model.safetensors.index.json".len()+SHARD.len()+3;
        let controls=Self::fixed_control_bytes();
        let canonical=canonical::resources::<std::io::Sink>(16).map_err(|_|Error::Bounds)?;
        // Simultaneous known+shape maps, moved/old-new Tensor slots, separate
        // header buffer + payload (header copy), index + config, three-file
        // pins/views, transition/cleanup controls and semantic hash workspace.
        // No body-copy buffer or serde::Value tree remains in this emitter.
        let mut resident=0usize;
        for n in [map_slots,known,tensor_slots,views,files,controls,header,index,payload,config,
            canonical.hash_workspace_bytes,size_of::<[StreamFrame<'_>;16]>()] {resident=plus(resident,n)?;}
        let file_bytes=plus(config,plus(index,payload)?)?;
        let comparisons=mul(mul(plus(ops.len(),descriptors)?,plus(ops.len(),descriptors)?)?,NAME)?;
        // Named logical passes: shape/tree/name comparisons; canonical semantic
        // cap; fixed-slice zero/init + JSON emission + header copy; BF16 stores;
        // complete THREE-file hashes; views/transition and whole-slot cleanup.
        let mut work=plus(CANONICAL_WORK,comparisons)?;
        for n in [resident,header,index,header,body,file_bytes,3*128,views,resident] {work=plus(work,n)?;}
        Ok(Self {descriptors,body,header,index,payload,resident,work:u64::try_from(work).map_err(|_|Error::Bounds)?})
    }
}

fn copy_text(value:&str) -> Result<String,Error> {
    let mut text=String::new(); text.try_reserve_exact(value.len()).map_err(|_|Error::Bounds)?;
    if text.capacity()>value.len() {return Err(Error::Bounds);} text.push_str(value); Ok(text)
}
fn emit(cap:usize, f:impl FnOnce(&mut Cursor<&mut [u8]>) -> std::io::Result<()>) -> Result<Vec<u8>,Error> {
    let mut bytes=exact_vec(cap)?; bytes.resize(cap,0);
    let len={let mut out=Cursor::new(bytes.as_mut_slice()); f(&mut out).map_err(|_|Error::Bounds)?; out.position() as usize};
    bytes.truncate(len); Ok(bytes)
}
fn json_string(out:&mut Cursor<&mut [u8]>, value:&str) -> std::io::Result<()> {
    serde_json::to_writer(out,value).map_err(std::io::Error::other)
}
fn emit_header(out:&mut Cursor<&mut [u8]>, tensors:&[Tensor]) -> std::io::Result<()> {
    out.write_all(b"{")?;
    for (i,t) in tensors.iter().enumerate() {
        if i!=0 {out.write_all(b",")?;} json_string(out,&t.name)?;
        write!(out,":{{\"data_offsets\":[{},{}],\"dtype\":",t.begin,t.end)?;
        json_string(out,&t.dtype)?; out.write_all(b",\"shape\":[")?;
        for (i,d) in t.shape.iter().enumerate() {if i!=0 {out.write_all(b",")?;} write!(out,"{d}")?;}
        out.write_all(b"]}")?;
    }
    out.write_all(b"}")
}
fn emit_index(out:&mut Cursor<&mut [u8]>, tensors:&[Tensor]) -> std::io::Result<()> {
    out.write_all(b"{\"weight_map\":{")?;
    for (i,t) in tensors.iter().enumerate() {
        if i!=0 {out.write_all(b",")?;} json_string(out,&t.name)?; out.write_all(b":")?; json_string(out,SHARD)?;
    }
    out.write_all(b"}}")
}

pub(super) const CONFIG: &[u8]=br#"{"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":3,"max_position_embeddings":16,"rms_norm_eps":0.000001,"hidden_act":"silu","tie_word_embeddings":false,"attention_bias":false,"rope_theta":10000.0}"#;
// Exact canonical output of the former serializer. No pre-owner Value/String/
// Vec factory. The equivalence regression below includes the imported protocol.
const EXPERIMENT:&[u8]=br#"{"budget":{"max_input_tokens":4,"max_new_tokens":2,"requests":1},"deployment":{"kind":"local","root":"."},"name":"raw-host-synthetic","pipeline":{"components":{"linear":{"component":"pllm/puma-three-party-decoder/v1","params":{"max_abs_activation":1,"range_policy":"checked","storage":"compact","weights":"public"}}},"model":{"source":"synthetic-original"}},"schema":"pllm.experiment.v1"}"#;
pub(super) fn experiment() -> &'static [u8] {EXPERIMENT}
#[derive(Clone)]
pub(super) struct Tensor {
    pub(super) name:String, pub(super) shape:Vec<usize>, pub(super) dtype:String,
    pub(super) begin:u64, pub(super) end:u64, pub(super) scale:Option<String>,
}
pub(super) struct Fixture {
    pub(super) experiment:&'static [u8], pub(super) files:Vec<(String,Vec<u8>)>,
    pub(super) sizes:Vec<u64>, pub(super) hashes:Vec<[u8;32]>,
    pub(super) tensors:Vec<Tensor>, pub(super) semantic:[u8;32], pub(super) recipe:String,
    #[cfg(test)] observe:Option<(OwnedBudget,usize)>,
    // LAST: all generated payload retires before its original covering charges.
    pub(super) seed: Option<SourceHeaderEntry>,
}
impl Fixture {
    pub(super) fn new() -> Self { Self::positive(CONFIG, false) }
    #[cfg(test)]
    pub(super) fn from_config(config: &[u8]) -> Self { Self::positive(config, false) }
    fn positive(config: &[u8], _export: bool) -> Self {
        Self::try_positive(config, _export).unwrap()
    }
    pub(super) fn try_positive(config: &[u8], _export: bool) -> Result<Self, Error> {
        let seed=SourceHeaderEntry::begin_mode(experiment(),config,4,2,[1;32],[2;32],[3;32],true)?;
        Self::from_seed(seed,_export)
    }
    fn from_seed(mut seed:SourceHeaderEntry,_export:bool) -> Result<Self,Error> {
        let quote=GenerationQuote::admit(&mut seed)?;
        // Reuse the existing callback boundary as well as its admission. Its
        // three collector control slots are already in the generation quote.
        let bytes=quote.resident.checked_sub(3*size_of::<SourceCollector<Self>>()).ok_or(Error::Bounds)?;
        let collected=seed.collect(0,bytes,0,|original,_|Self::generate(original,quote))?;
        let SourceCollector {payload,header}=collected;
        let mut fixture=payload.ok_or(Error::Binding)?;
        // Infallible ownership transfer, never a replacement header or guard.
        fixture.seed=Some(header);
        Ok(fixture)
    }
    fn generate(original:&OriginalSourcePlan, q:GenerationQuote) -> Result<Self,Error> {
        generation_event(0);
        // The shared collector retains the original cover until this callback
        // returns, including destruction of every partially built payload.
        let mut fixture=Self {experiment:experiment(),files:exact_vec(3)?,sizes:exact_vec(3)?,hashes:exact_vec(3)?,
            tensors:exact_vec(q.descriptors)?,semantic:[0;32],recipe:copy_text("D31")?,
            #[cfg(test)] observe:None,
            seed:None};
        let (shapes,_)=crate::replicated_decoder::program::source_shapes_from_plan(original.plan(),WeightMode::Public).map_err(|_|Error::Binding)?;
        if shapes.len()>=q.descriptors || shapes.contains_key("zz_unused.weight") {return Err(Error::Bounds);}
        generation_event(3); fixture.semantic=semantic_identity(original.plan())?;
        let mut end=0usize;
        for (name,shape) in shapes.into_iter().chain(std::iter::once((copy_text("zz_unused.weight")?,vec![3,2]))) {
            let count=shape.iter().try_fold(1usize,|n,d|mul(n,*d))?;
            let begin=end; end=plus(end,mul(2,count)?)?;
            if end>q.body || fixture.tensors.len()==q.descriptors || name.capacity()>NAME || shape.capacity()>2 {return Err(Error::Bounds);}
            generation_event(1);
            fixture.tensors.push(Tensor {name,shape,dtype:copy_text("BF16")?,begin:begin as u64,end:end as u64,scale:None});
        }
        let json=emit(q.header,|out|emit_header(out,&fixture.tensors))?;
        let index=emit(q.index,|out|emit_index(out,&fixture.tensors))?;
        let mut payload=exact_vec(q.payload)?;
        payload.extend_from_slice(&(json.len() as u64).to_le_bytes()); payload.extend_from_slice(&json);
        generation_event(2);
        let words=[0x3f80u16,0x4000,0x4080,0x4100,0x3f00,0x3e80];
        for tensor in &fixture.tensors {
            for i in 0..(tensor.end-tensor.begin)/2 {payload.extend_from_slice(&words[i as usize%words.len()].to_le_bytes());}
        }
        if payload.capacity()>q.payload || payload.len()>q.payload || json.capacity()>q.header || index.capacity()>q.index {return Err(Error::Bounds);}
        let mut config=exact_vec(original.raw_configuration().len())?;
        config.extend_from_slice(original.raw_configuration());
        for (name,bytes) in [("config.json",config),("model.safetensors.index.json",index),(SHARD,payload)] {
            fixture.sizes.push(bytes.len() as u64); generation_event(3);
            fixture.hashes.push(Sha256::digest(&bytes).into()); fixture.files.push((copy_text(name)?,bytes));
        }
        Ok(fixture)
    }
    // Generation reserves ONE view construction for the consuming handoff,
    // not arbitrary repeated borrowed calls. Test callers must fund extra use.
    pub(super) fn with_expected<T>(&self,f:impl FnOnce(HostRawExpectations<'_>)->T) -> Result<T,Error> {
        let mut files=exact_vec(self.files.len())?;
        for (i,(name,_)) in self.files.iter().enumerate() {
            files.push((name.as_str(),*self.sizes.get(i).ok_or(Error::Binding)?,*self.hashes.get(i).ok_or(Error::Binding)?));
        }
        let mut tensors=exact_vec(self.tensors.len())?;
        for t in &self.tensors {tensors.push((t.name.as_str(),2,t.dtype.as_str(),t.shape.as_slice(),t.begin,t.end,t.scale.as_deref()));}
        Ok(f(HostRawExpectations::provision(&files,&tensors,*self.hashes.first().ok_or(Error::Binding)?,self.semantic,&self.recipe)))
    }
    pub(super) fn header_result(self) -> Result<RawPending,Error> {
        self.into_collector()?.into_fixture_raw_host()
    }
    pub(super) fn into_collector(mut self) -> Result<SourceCollector<Self>,Error> {
        // Complete every fallible metadata check while self still owns cover.
        let bytes=self.seed.as_ref().ok_or(Error::Binding)?.collection_charge.payload_bytes()
            .checked_sub(3*size_of::<SourceCollector<Self>>()).ok_or(Error::Bounds)?;
        let header=self.seed.take().ok_or(Error::Binding)?;
        // Shared collector's payload-before-header ordering also covers failure
        // in trusted provisioning, configuration parsing and directory setup.
        // Re-enter its existing zero-additional-work callback without a guard
        // adoption constructor or a refund of any retained collection charge.
        header.collect(0,bytes,0,move|_,_|Ok(self))
    }
    #[cfg(test)]
    pub(super) fn header(self) -> RawPending { self.header_result().unwrap() }

    // Test-only positive FP8/long-pin input variations. The existing generation
    // debit remains: reserve new descriptor/name/view + fixed-emission buffers
    // AND the temporary body before editing or copying any original bytes.
    #[cfg(test)]
    pub(super) fn reserve_edit(&mut self,descriptors:usize,body:usize) -> Result<(),Error> {
        if descriptors>source_policy::DESCRIPTORS {return Err(Error::Bounds);}
        let names=mul(descriptors,NAME)?;
        let record=r#""":{"data_offsets":[,],"dtype":"F8_E4M3","shape":[,]},"#.len()+80;
        let h=plus(2,plus(mul(6,names)?,mul(descriptors,record)?)?)?;
        let i=plus(r#"{"weight_map":{}}"#.len(),plus(mul(6,names)?,mul(descriptors,SHARD.len()+6)?)?)?;
        let p=plus(8,plus(h,body)?)?;
        let slots=mul(descriptors,size_of::<Tensor>()+2*NAME+2*size_of::<usize>()+8+size_of::<TensorView<'_>>())?;
        let mut resident=plus(slots,4*size_of::<Sha256>()+3*size_of::<Fixture>())?;
        for n in [h,i,p,body,2*NAME,6] {resident=plus(resident,n)?;}
        let work=plus(mul(2,resident)?,plus(plus(p,i)?,mul(mul(2,descriptors)?,mul(descriptors,NAME)?)?)?)?;
        let seed=self.seed.as_mut().ok_or(Error::Binding)?;
        seed.admit_collection(0,plus(seed.collection_charge.payload_bytes(),resident)?,work as u64)?;
        self.tensors.try_reserve_exact(descriptors.saturating_sub(self.tensors.len())).map_err(|_|Error::Bounds)?;
        if self.tensors.capacity()>descriptors {return Err(Error::Bounds);} Ok(())
    }
    #[cfg(test)]
    pub(super) fn reemit(&mut self,body:Vec<u8>) -> Result<(),Error> {
        let m=self.tensors.len(); let names=self.tensors.iter().try_fold(0usize,|n,t|plus(n,t.name.len()))?;
        let record=r#""":{"data_offsets":[,],"dtype":"F8_E4M3","shape":[,]},"#.len()+80;
        let h=plus(2,plus(mul(6,names)?,mul(m,record)?)?)?;
        let i=plus(r#"{"weight_map":{}}"#.len(),plus(mul(6,names)?,mul(m,SHARD.len()+6)?)?)?;
        let json=emit(h,|out|emit_header(out,&self.tensors))?;
        let index=emit(i,|out|emit_index(out,&self.tensors))?;
        let mut payload=exact_vec(plus(8,plus(h,body.len())?)?)?;
        payload.extend_from_slice(&(json.len() as u64).to_le_bytes()); payload.extend_from_slice(&json); payload.extend_from_slice(&body);
        self.files[2].1=payload; self.files[1].1=index;
        for n in [1,2] {generation_event(3); self.sizes[n]=self.files[n].1.len() as u64; self.hashes[n]=Sha256::digest(&self.files[n].1).into();}
        Ok(())
    }
}

#[cfg(test)]
impl Drop for Fixture {
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.files)); drop(std::mem::take(&mut self.tensors));
        drop(std::mem::take(&mut self.sizes)); drop(std::mem::take(&mut self.hashes));
        drop(std::mem::take(&mut self.recipe));
        if let Some((counter,cover))=&self.observe {
            GENERATION_DROP.with(|n|n.set(Some((self.files.capacity()==0 && self.tensors.capacity()==0,counter.budget().used_bytes(),*cover))));
        }
        // All other owned fields are empty; observe/seed retire afterwards.
    }
}

impl SourceCollector<Fixture> {
    fn into_fixture_raw_host(mut self) -> Result<RawPending,Error> {
        // Provision while the generated payload AND original charges are in
        // this payload-first collector. On error it destroys payload first.
        let fixture=self.payload.as_ref().ok_or(Error::Binding)?;
        let directory=fixture.with_expected(|e|self.header.prepare_raw_host(e))??;
        let SourceCollector {payload,header}=self;
        // No fallible operation after splitting the collector.
        let header=Arc::new(header); let identity=Arc::downgrade(&header);
        Ok(RawPending {fixture:payload,header,identity,directory})
    }
}

#[cfg(test)]
mod funding_tests {
    use super::*;
    fn seed(config:&[u8]) -> SourceHeaderEntry {
        SourceHeaderEntry::begin_mode(experiment(),config,4,2,[1;32],[2;32],[3;32],true).unwrap()
    }
    // Literal fixed type/slot floor, independent of the full resident formula.
    // This is logical control accounting, not allocator/stack measurement.
    fn bootstrap_control_floor() -> usize {
        size_of::<[GenerationQuote;3]>()
            +size_of::<[SourceCollector<Fixture>;3]>()
            +size_of::<[RawPending;3]>()
            +size_of::<[HostRawExpectations<'_>;3]>()
            +size_of::<[Sha256;4]>()
            +size_of::<Cursor<&mut [u8]>>()
            +size_of::<serde_json::Serializer<Cursor<&mut [u8]>>>()
            +size_of::<std::collections::BTreeMap<String,Vec<usize>>>()
            +size_of::<std::collections::BTreeMap<String,usize>>()
            +size_of::<[(&str,Vec<u8>);3]>()
            +size_of::<[[u8;32];3]>()
    }
    #[test]
    fn quotation_bootstrap_matches_literal_fixed_control_floor() {
        assert_eq!(GenerationQuote::fixed_control_bytes(),bootstrap_control_floor());
    }
    #[test]
    fn quotation_bootstrap_exact_and_one_short_before_metadata_retains_original_debits_and_excess() {
        let bootstrap=bootstrap_control_floor();
        for excess in [0,2*bootstrap+17] {for short in [false,true] {
            let mut header=seed(CONFIG);
            let existing=header.collection_charge.payload_bytes()+excess;
            header.admit_collection(0,existing,23).unwrap(); // real prior work/excess
            let lower_bytes=header.original.lower_bootstrap_bytes(header.limits).unwrap();
            let sibling=header.original.lower(header.root.charge_external(lower_bytes).unwrap(),header.original.workload(),header.limits).unwrap();
            assert!(sibling.same_origin(&header.original));
            let control=header.root.charge_external(OriginalSourcePlan::compiler_metadata_control_bytes().unwrap()).unwrap();
            let n=header.original.plan().prefill.operations.len();
            let census=(CANONICAL_WORK+4*(n+1)*(n+1)*NAME) as u64;
            // Leave census plus ONE on the actual original ledger. Short bytes
            // must retain that debit; exact bytes allow metadata, then the
            // separate generation-work admission must refuse rather than emit.
            header.original.prepay_compiler_metadata(&header.controller,
                header.limits.max_work-sibling.resources().prepaid_work-census-1).unwrap();
            let counter=header.root.try_clone().unwrap();
            let plan=header.original.plan() as *const _;
            let mut outside=header.root.charge_external(0).unwrap();
            outside.resize_payload(MAX_PROGRAM_BYTES-counter.budget().used_bytes()-bootstrap+usize::from(short)).unwrap();
            let live=counter.budget().used_bytes();
            let visits=METADATA_VISITS.with(|n|n.get());
            let events=GENERATION_EVENTS.with(|n|n.get());
            assert!(GenerationQuote::admit(&mut header).is_err());
            assert!(header.failed);
            assert_eq!(METADATA_VISITS.with(|n|n.get()),visits+usize::from(!short));
            assert_eq!(GENERATION_EVENTS.with(|n|n.get()),events);
            let growth=if short {0} else {bootstrap};
            assert_eq!(header.collection_charge.payload_bytes(),existing+growth);
            assert_eq!(counter.budget().used_bytes(),live+growth);
            assert!(std::ptr::eq(counter.budget(),header.original.budget()));
            assert!(std::ptr::eq(plan,header.original.plan()));
            assert!(sibling.same_origin(&header.original));
            if short {
                // Byte refusal is not an immediate work-ledger burn. Exactly
                // ONE remains: spending it proves the census was not refunded.
                sibling.prepay_compiler_metadata(&control,0).unwrap();
                sibling.prepay_compiler_metadata(&control,1).unwrap();
                assert!(sibling.prepay_compiler_metadata(&control,1).is_err());
            } else {
                // The later generation-work refusal burns this same ledger.
                assert!(sibling.prepay_compiler_metadata(&control,0).is_err());
            }
            assert!(header.admit_collection(0,existing,0).is_err());
            drop(outside); drop(header); drop(sibling); drop(control);
            assert_eq!(counter.budget().used_bytes(),OwnedBudget::resources().unwrap().initial_reserved_bytes);
        }}
    }
    // Independent literal geometry of CONFIG: two 3x2 embedding/head matrices,
    // seven 2x2 projections, three length-two norms AND three Qwen2 Q/K/V
    // biases (the existing Qwen2 adapter emits them), unused 3x2. No candidate
    // tensors, generated bytes, shape map or GenerationQuote supplies geometry.
    // The remaining formula agreement is NOT an independent proof that the
    // whole layout inventory is complete.
    fn tiny_oracle(seed:&SourceHeaderEntry) -> (usize,u64,u64) {
        let names=["lm_head.weight","model.embed_tokens.weight","model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.down_proj.weight","model.layers.0.mlp.gate_proj.weight","model.layers.0.mlp.up_proj.weight",
            "model.layers.0.post_attention_layernorm.weight","model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.o_proj.weight","model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.v_proj.weight","model.norm.weight","zz_unused.weight",
            "model.layers.0.self_attn.q_proj.bias","model.layers.0.self_attn.k_proj.bias","model.layers.0.self_attn.v_proj.bias"];
        let m=names.len(); let a:usize=names.iter().map(|v|v.len()).sum(); let n=seed.original.plan().prefill.operations.len();
        let b=2*(2*3*2+7*2*2+3*2+3*2+3*2);
        let h=2+6*a+m*(r#""":{"data_offsets":[,],"dtype":"BF16","shape":[,]},"#.len()+80);
        let i=r#"{"weight_map":{}}"#.len()+6*a+m*("synthetic-00001.safetensors".len()+6);
        let p=8+h+b;
        let map=m*(source_registry::tree_entry_bytes::<(String,Vec<usize>)>().unwrap()+2*size_of::<usize>())+a;
        let known=n*(source_registry::tree_entry_bytes::<(String,usize)>().unwrap()+NAME);
        let tensors=m*(size_of::<Tensor>()+2*size_of::<usize>()+4)+a;
        let views=3*size_of::<(&str,u64,[u8;32])>()+m*size_of::<(&str,usize,&str,&[usize],u64,u64,Option<&str>)>();
        let files=3*(size_of::<(String,Vec<u8>)>()+size_of::<u64>()+size_of::<[u8;32]>())
            +"config.json".len()+"model.safetensors.index.json".len()+"synthetic-00001.safetensors".len()+3;
        let controls=3*size_of::<SourceCollector<Fixture>>()+3*size_of::<RawPending>()+3*size_of::<HostRawExpectations<'_>>()
            +4*size_of::<Sha256>()+size_of::<Cursor<&mut [u8]>>()+size_of::<serde_json::Serializer<Cursor<&mut [u8]>>>()
            +size_of::<std::collections::BTreeMap<String,Vec<usize>>>()+size_of::<std::collections::BTreeMap<String,usize>>()
            +3*size_of::<GenerationQuote>()+size_of::<[(&str,Vec<u8>);3]>()+3*size_of::<[u8;32]>();
        let c=CONFIG.len();
        let resident=map+known+tensors+views+files+controls+h+i+p+c
            +canonical::resources::<std::io::Sink>(16).unwrap().hash_workspace_bytes+size_of::<[StreamFrame<'_>;16]>();
        let work=CANONICAL_WORK+(n+m)*(n+m)*NAME+resident+h+i+h+b+(c+i+p)+3*128+views+resident;
        (resident,work as u64,(CANONICAL_WORK+4*(n+1)*(n+1)*NAME) as u64)
    }
    #[test]
    fn full_generation_target_retains_existing_excess_and_counts_bootstrap_once() {
        for export in [false,true] {
            let mut header=seed(CONFIG);
            let existing=header.collection_charge.payload_bytes()+23;
            header.admit_collection(0,existing,17).unwrap();
            let (resident,_,_)=tiny_oracle(&header); // formula agreement only
            let counter=header.root.try_clone().unwrap();
            let mut outside=header.root.charge_external(0).unwrap();
            outside.resize_payload(MAX_PROGRAM_BYTES-counter.budget().used_bytes()-resident).unwrap();
            let fixture=Fixture::from_seed(header,export).unwrap();
            let retained=fixture.seed.as_ref().unwrap();
            assert!(std::ptr::eq(counter.budget(),retained.original.budget()));
            assert_eq!(retained.collection_charge.payload_bytes(),existing+resident);
            assert_eq!(counter.budget().used_bytes(),MAX_PROGRAM_BYTES);
            drop(fixture); drop(outside);
            assert_eq!(counter.budget().used_bytes(),OwnedBudget::resources().unwrap().initial_reserved_bytes);
        }
    }
    #[test]
    fn static_experiment_equals_existing_serializer() {
        let prior=serde_json::to_vec(&serde_json::json!({"schema":"pllm.experiment.v1","name":"raw-host-synthetic","pipeline":{"model":{"source":"synthetic-original"},"components":{"linear":{"component":crate::replicated_decoder::sdk::PROTOCOL,"params":{"weights":"public","range_policy":"checked","max_abs_activation":1,"storage":"compact"}}}},"deployment":{"kind":"local","root":"."},"budget":{"requests":1,"max_input_tokens":4,"max_new_tokens":2}})).unwrap();
        assert_eq!(experiment(),prior);
    }
    #[test]
    fn generation_exact_and_one_short_work_on_retained_original_burns_sibling() {
        for export in [false,true] {for short in [false,true] {
            let mut header=seed(CONFIG); let (bytes,work,census)=tiny_oracle(&header);
            header.original.prepay_compiler_metadata(&header.controller,17).unwrap(); // actual prior debit
            let bootstrap=header.original.lower_bootstrap_bytes(header.limits).unwrap();
            let sibling=header.original.lower(header.root.charge_external(bootstrap).unwrap(),header.original.workload(),header.limits).unwrap();
            assert!(sibling.same_origin(&header.original));
            let used=sibling.resources().prepaid_work;
            header.original.prepay_compiler_metadata(&header.controller,header.limits.max_work-used-work-census+u64::from(short)).unwrap();
            let control=header.root.charge_external(OriginalSourcePlan::compiler_metadata_control_bytes().unwrap()).unwrap();
            let counter=header.root.try_clone().unwrap(); let before=GENERATION_EVENTS.with(|n|n.get());
            // SAME admission called by both export selections; no helper ledger.
            let result=Fixture::from_seed(header,export);
            if short {
                assert!(result.is_err()); assert_eq!(GENERATION_EVENTS.with(|n|n.get()),before);
                assert!(sibling.prepay_compiler_metadata(&control,0).is_err());
            } else {
                let fixture=result.unwrap();
                let original=fixture.seed.as_ref().unwrap();
                assert_eq!(original.collection_charge.payload_bytes(),bytes);
                assert!(std::ptr::eq(original.original.budget(),counter.budget()));
                sibling.prepay_compiler_metadata(&original.controller,0).unwrap();
                assert!(sibling.prepay_compiler_metadata(&original.controller,1).is_err());
                let live=counter.budget().used_bytes(); drop(fixture); assert!(counter.budget().used_bytes()<live);
            }
        }}
    }
    #[test]
    fn generation_exact_and_one_short_bytes_before_all_generation_factories() {
        for export in [false,true] {for short in [false,true] {
            let header=seed(CONFIG); let (bytes,work,census)=tiny_oracle(&header);
            let bootstrap=header.original.lower_bootstrap_bytes(header.limits).unwrap();
            let sibling=header.original.lower(header.root.charge_external(bootstrap).unwrap(),header.original.workload(),header.limits).unwrap();
            assert!(sibling.same_origin(&header.original));
            header.original.prepay_compiler_metadata(&header.controller,
                header.limits.max_work-sibling.resources().prepaid_work-work-census).unwrap();
            let control=header.root.charge_external(OriginalSourcePlan::compiler_metadata_control_bytes().unwrap()).unwrap();
            let counter=header.root.try_clone().unwrap(); let mut outside=header.root.charge_external(0).unwrap();
            outside.resize_payload(MAX_PROGRAM_BYTES-counter.budget().used_bytes()-bytes+usize::from(short)).unwrap();
            let before=GENERATION_EVENTS.with(|n|n.get()); let result=Fixture::from_seed(header,export);
            assert_eq!(result.is_err(),short);
            if short {assert_eq!(GENERATION_EVENTS.with(|n|n.get()),before);} else {
                let fixture=result.unwrap();
                assert_eq!(counter.budget().used_bytes(),MAX_PROGRAM_BYTES);
                drop(fixture); assert!(counter.budget().used_bytes()<MAX_PROGRAM_BYTES);
            }
            // Byte refusal cannot refund paid census/generation passes. This is
            // an actual pre-existing same-origin sibling, not a helper meter.
            // Byte refusal alone does not burn a zero-work sibling; the later
            // explicit quota overrun below does.
            sibling.prepay_compiler_metadata(&control,0).unwrap();
            assert!(sibling.prepay_compiler_metadata(&control,1).is_err());
            assert!(sibling.prepay_compiler_metadata(&control,0).is_err());
        }}
    }
    #[test]
    fn census_one_short_burns_original_sibling_before_metadata_or_generation() {
        for export in [false,true] {
            let header=seed(CONFIG); let (_,_,census)=tiny_oracle(&header);
            let bootstrap=header.original.lower_bootstrap_bytes(header.limits).unwrap();
            let sibling=header.original.lower(header.root.charge_external(bootstrap).unwrap(),header.original.workload(),header.limits).unwrap();
            assert!(sibling.same_origin(&header.original));
            header.original.prepay_compiler_metadata(&header.controller,
                header.limits.max_work-sibling.resources().prepaid_work-census+1).unwrap();
            let control=header.root.charge_external(OriginalSourcePlan::compiler_metadata_control_bytes().unwrap()).unwrap();
            let before=GENERATION_EVENTS.with(|n|n.get());
            assert!(Fixture::from_seed(header,export).is_err());
            assert_eq!(GENERATION_EVENTS.with(|n|n.get()),before);
            assert!(sibling.prepay_compiler_metadata(&control,0).is_err());
        }
    }
    #[test]
    fn malformed_configuration_refuses_before_generation() {
        for export in [false,true] {
            let before=GENERATION_EVENTS.with(|n|n.get());
            assert!(Fixture::try_positive(b"not JSON",export).is_err());
            assert_eq!(GENERATION_EVENTS.with(|n|n.get()),before);
        }
    }
    #[test]
    fn bounded_emission_matches_old_format_and_modes_retain_actual_original() {
        let a=Fixture::try_positive(CONFIG,false).unwrap(); let b=Fixture::try_positive(CONFIG,true).unwrap();
        assert_eq!(a.files,b.files); assert_eq!(a.hashes,b.hashes);
        assert_eq!(a.tensors.len(),16); // literal Qwen2 inventory, including Q/K/V biases
        let emitted_header=u64::from_le_bytes(a.files[2].1[..8].try_into().unwrap()) as usize;
        assert_eq!(a.files[2].1.len()-8-emitted_header,2*(2*3*2+7*2*2+3*2+3*2+3*2));
        let mut header=serde_json::Map::new(); let mut map=serde_json::Map::new();
        for t in &a.tensors {
            header.insert(t.name.clone(),serde_json::json!({"dtype":t.dtype,"shape":t.shape,"data_offsets":[t.begin,t.end]}));
            map.insert(t.name.clone(),serde_json::json!(SHARD));
        }
        let h=serde_json::to_vec(&header).unwrap();
        assert_eq!(&a.files[2].1[8..8+h.len()],h);
        assert_eq!(a.files[1].1,serde_json::to_vec(&serde_json::json!({"weight_map":map})).unwrap());
        for fixture in [a,b] {
            let original=fixture.seed.as_ref().unwrap(); let counter=original.root.try_clone().unwrap();
            let plan=original.original.plan() as *const _;
            let config=original.original.raw_configuration().as_ptr();
            let bootstrap=original.original.lower_bootstrap_bytes(original.limits).unwrap();
            let sibling=original.original.lower(original.root.charge_external(bootstrap).unwrap(),original.original.workload(),original.limits).unwrap();
            let before=sibling.resources().prepaid_work;
            let pending=fixture.header_result().unwrap();
            assert!(std::ptr::eq(counter.budget(),pending.original.budget())); assert!(sibling.same_origin(&pending.original));
            assert!(std::ptr::eq(plan,pending.original.plan())); assert_eq!(config,pending.original.raw_configuration().as_ptr());
            assert!(pending.fixture.as_ref().unwrap().seed.is_none());
            let bootstrap=pending.original.lower_bootstrap_bytes(pending.limits).unwrap();
            let after=pending.original.lower(pending.root.charge_external(bootstrap).unwrap(),pending.original.workload(),pending.limits).unwrap();
            assert!(after.resources().prepaid_work>before); // transition debits original, not a reset
        }
    }
    #[test]
    fn general_nondefault_geometry_keeps_positive_factory() {
        let mut config:serde_json::Value=serde_json::from_slice(CONFIG).unwrap();
        config["hidden_size"]=serde_json::json!(4); config["intermediate_size"]=serde_json::json!(6);
        let config=serde_json::to_vec(&config).unwrap();
        for export in [false,true] {
            let fixture=Fixture::try_positive(&config,export).unwrap();
            assert_eq!(fixture.tensors.iter().find(|t|t.name=="model.embed_tokens.weight").unwrap().shape,[3,4]);
            assert_eq!(fixture.tensors.iter().find(|t|t.name=="model.layers.0.mlp.gate_proj.weight").unwrap().shape,[6,4]);
            let h=u64::from_le_bytes(fixture.files[2].1[..8].try_into().unwrap()) as usize;
            // Independent two embedding/head, four attention, three MLP, three
            // norm, three Q/K/V bias and one unused tensor products.
            assert_eq!(fixture.files[2].1.len()-8-h,2*(2*3*4+4*4*4+3*6*4+3*4+3*4+3*2));
            fixture.header_result().unwrap();
        }
    }
    #[test]
    fn failed_consuming_transition_drops_payload_before_original_cover() {
        let mut fixture=Fixture::new(); let seed=fixture.seed.as_ref().unwrap();
        let counter=seed.root.try_clone().unwrap(); let cover=seed.collection_charge.charged_bytes();
        fixture.observe=Some((counter.try_clone().unwrap(),cover)); fixture.semantic[0]^=1;
        GENERATION_DROP.with(|n|n.set(None)); let live=counter.budget().used_bytes();
        assert!(fixture.header_result().is_err());
        let (empty,used,expected)=GENERATION_DROP.with(|n|n.get()).unwrap();
        assert!(empty); assert_eq!(expected,cover); assert!(used>=cover); assert!(used>=live);
        assert_eq!(counter.budget().used_bytes(),OwnedBudget::resources().unwrap().initial_reserved_bytes);
    }
}

/// Authorized synthetic preparation experiment ONLY. Not downstream ingestion,
/// an admitted RAW owner, a source-funding exemption or execution evidence.
/// Keep this exporter unit-only; literal adoption requires separate review.
/// Generation admission does not fund the subsequent export experiment's
/// validation, parsing, formatting or output.
#[cfg(test)]
#[test]
#[ignore = "explicit D55 synthetic preparation; separate execution grant required"]
fn export_d55_positive_fixture_literals() {
    use std::fmt::Write as _;
    use std::io::Write as _;

    const MAX_FILE_BYTES: usize = 32 * 1024;
    const MAX_METADATA_BYTES: usize = 4096;
    const MAX_DESCRIPTORS: usize = 64;
    const MAX_NAME_BYTES: usize = 128;
    const MAX_EXPORT_BYTES: usize = 512 * 1024;
    // Both selections use the SAME general, original-ledger admission. These
    // additional export-only checks do not substitute for construction funding.
    let fixture = Fixture::positive(CONFIG, true);
    assert_eq!(fixture.recipe, "D31");
    assert_eq!(fixture.files.len(), 3);
    assert_eq!(fixture.sizes.len(), 3);
    assert_eq!(fixture.hashes.len(), 3);
    assert_eq!(fixture.files[0].1, CONFIG);
    assert!(CONFIG.len() <= MAX_METADATA_BYTES);
    assert!(fixture.experiment.len() <= MAX_METADATA_BYTES);
    assert!(!fixture.tensors.is_empty() && fixture.tensors.len() <= MAX_DESCRIPTORS);
    let total = fixture.files.iter().try_fold(0usize, |n, (_, bytes)| n.checked_add(bytes.len())).unwrap();
    assert!(total <= MAX_FILE_BYTES);
    let bounded_name = |name: &str| {
        name.len() <= MAX_NAME_BYTES && name.bytes().all(|byte| (0x20..=0x7e).contains(&byte))
    };
    for (index, (name, bytes)) in fixture.files.iter().enumerate() {
        assert!(bounded_name(name));
        assert_eq!(fixture.sizes[index], bytes.len() as u64);
        let actual: [u8; 32] = Sha256::digest(bytes).into();
        assert_eq!(fixture.hashes[index], actual);
    }
    let shard = &fixture.files[2].1;
    let header_length = usize::try_from(u64::from_le_bytes(shard.get(..8).unwrap().try_into().unwrap())).unwrap();
    let body_start = 8usize.checked_add(header_length).unwrap();
    let body_length = u64::try_from(shard.len().checked_sub(body_start).unwrap()).unwrap();
    let header: serde_json::Value = serde_json::from_slice(shard.get(8..body_start).unwrap()).unwrap();
    let header = header.as_object().unwrap();
    let index: serde_json::Value = serde_json::from_slice(&fixture.files[1].1).unwrap();
    let index = index["weight_map"].as_object().unwrap();
    assert_eq!(header.len(), fixture.tensors.len());
    assert_eq!(index.len(), fixture.tensors.len());
    let mut names = std::collections::BTreeSet::new();
    let mut cursor = 0u64;
    for tensor in &fixture.tensors {
        assert!(bounded_name(&tensor.name));
        assert_eq!(tensor.dtype, "BF16");
        assert!(tensor.scale.is_none());
        assert!((1..=2).contains(&tensor.shape.len()));
        assert!(tensor.shape.iter().all(|dimension| (1..=16).contains(dimension)));
        assert!(names.insert(tensor.name.as_str()), "duplicate descriptor name");
        assert!(tensor.begin <= tensor.end && tensor.end <= body_length);
        let words = tensor.shape.iter().try_fold(1u64, |n, &d| n.checked_mul(u64::try_from(d).unwrap())).unwrap();
        assert_eq!(tensor.end.checked_sub(tensor.begin).unwrap(), words.checked_mul(2).unwrap());
        assert_eq!(tensor.begin, cursor);
        cursor = tensor.end;
        let record = header.get(&tensor.name).unwrap().as_object().unwrap();
        assert_eq!(record.len(), 3);
        assert_eq!(record["dtype"].as_str(), Some("BF16"));
        let shape = record["shape"].as_array().unwrap();
        assert_eq!(shape.len(), tensor.shape.len());
        for (actual, expected) in shape.iter().zip(&tensor.shape) {
            assert_eq!(actual.as_u64(), Some(u64::try_from(*expected).unwrap()));
        }
        let offsets = record["data_offsets"].as_array().unwrap();
        assert_eq!(offsets.len(), 2);
        assert_eq!(offsets[0].as_u64(), Some(tensor.begin));
        assert_eq!(offsets[1].as_u64(), Some(tensor.end));
        assert_eq!(index.get(&tensor.name).unwrap().as_str(), Some(fixture.files[2].0.as_str()));
    }
    assert_eq!(cursor, body_length);
    assert_eq!(fixture.files[0].0, "config.json");
    assert_eq!(fixture.files[1].0, "model.safetensors.index.json");
    assert_eq!(fixture.files[2].0, "synthetic-00001.safetensors");

    // All checks precede formatting/output. Debug byte literals cost <=5 bytes
    // per input byte; bounded ASCII names cost <=6 bytes per character. With
    // <=64 rank-two descriptors and fixed-size digests/pins, output stays below
    // 512 KiB, reserving the other half of the 1 MiB log cap for test framing.
    let mut output = String::new();
    writeln!(output, "// D55 synthetic preparation literals; review before adoption.").unwrap();
    writeln!(output, "pub const RECIPE: &str = \"D31\";").unwrap();
    writeln!(output, "pub const ROWS: usize = 4;\npub const OUTPUTS: usize = 2;").unwrap();
    writeln!(output, "pub const SESSION: [u8; 32] = {:?};", [1u8; 32]).unwrap();
    writeln!(output, "pub const SOURCE: [u8; 32] = {:?};", [2u8; 32]).unwrap();
    writeln!(output, "pub const COMPOSITION: [u8; 32] = {:?};", [3u8; 32]).unwrap();
    writeln!(output, "pub const DYADIC_BF16_WORDS: [u16; 6] = {:?};", [0x3f80u16, 0x4000, 0x4080, 0x4100, 0x3f00, 0x3e80]).unwrap();
    writeln!(output, "pub const CONFIG_BYTES: &[u8] = &{:?};", CONFIG).unwrap();
    writeln!(output, "pub const CONFIG_LENGTH: usize = {};", CONFIG.len()).unwrap();
    writeln!(output, "pub const CONFIG_SHA256: [u8; 32] = {:?};", fixture.hashes[0]).unwrap();
    writeln!(output, "pub const EXPERIMENT_BYTES: &[u8] = &{:?};", fixture.experiment).unwrap();
    writeln!(output, "pub const EXPERIMENT_LENGTH: usize = {};", fixture.experiment.len()).unwrap();
    let experiment_sha: [u8; 32] = Sha256::digest(&fixture.experiment).into();
    writeln!(output, "pub const EXPERIMENT_SHA256: [u8; 32] = {:?};", experiment_sha).unwrap();
    for (index, (_, bytes)) in fixture.files.iter().enumerate() {
        writeln!(output, "pub const FILE_{index}_BYTES: &[u8] = &{:?};", bytes).unwrap();
    }
    writeln!(output, "pub const FILES: &[(&str, u64, [u8; 32])] = &[").unwrap();
    for (index, (name, _)) in fixture.files.iter().enumerate() {
        writeln!(output, "    ({:?}, {}, {:?}),", name, fixture.sizes[index], fixture.hashes[index]).unwrap();
    }
    writeln!(output, "];\npub const FILE_COUNT: usize = 3;").unwrap();
    writeln!(output, "pub const TENSORS: &[(&str, usize, &str, &[usize], u64, u64, Option<&str>)] = &[").unwrap();
    for tensor in &fixture.tensors {
        writeln!(output, "    ({:?}, 2, {:?}, &{:?}, {}, {}, None),", tensor.name, tensor.dtype, tensor.shape, tensor.begin, tensor.end).unwrap();
    }
    writeln!(output, "];\npub const DESCRIPTOR_COUNT: usize = {};", fixture.tensors.len()).unwrap();
    writeln!(output, "pub const SEMANTIC_EXPECTATION: [u8; 32] = {:?};", fixture.semantic).unwrap();
    assert!(output.len() <= MAX_EXPORT_BYTES, "D55 export exceeds finite output bound");
    // No fixture.header_result(), RawHost factory, append, finish, reader,
    // retirement or Python integration is called by this export case.
    std::io::stdout().lock().write_all(output.as_bytes()).unwrap();
}
