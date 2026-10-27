//! Source-owned clear-oracle prefix only, not CLEAR custody or a model route.
//! The gamma reservation retains the actual lookup, request, reader and original
//! immutable plan. No slice, equality label or budget membership constructs it.
use super::*;

// Explicit identity for the already approved fixed-epsilon/cast-before-gamma
// arithmetic. Not selected by model family, and not a new default dispatch.
pub(super) const PROFILE: &str = "bf16-rn32-rms-epsilon-358637bd-cast-before-direct-gamma";

/// Output cannot escape independently of its original source-owned prefix.
pub(in crate::replicated_decoder) struct RawFirstNormActivation {
    words: Vec<u16>,
    norm: AdmittedRawNormSpec, // gamma/lookup/source drop AFTER output wipe/free
}
impl Drop for RawFirstNormActivation {
    fn drop(&mut self) {
        #[cfg(test)]
        OUTPUT_PREWIPE.with(|n|n.set(Some((self.words.len(),self.words.capacity(),
            self.words.iter().any(|w|*w!=0)))));
        self.words.resize(self.words.capacity(),0);
        self.words.fill(0);
        #[cfg(test)]
        let wiped=self.words.iter().all(|w|*w==0);
        drop(std::mem::take(&mut self.words));
        #[cfg(test)]
        OUTPUT_DROP.with(|n|n.set(Some((wiped && self.words.capacity()==0,
            self.norm.reservation.charge.budget().used_bytes()))));
    }
}
impl RawFirstNormActivation {
    pub(in crate::replicated_decoder) fn words(&self) -> &[u16] { &self.words }
    pub(in crate::replicated_decoder) fn shape(&self) -> [usize;3] { self.norm.capability.shape }
    pub(in crate::replicated_decoder) fn check_source(&self, host: &RawHost) -> Result<(),Error> {
        self.norm.reservation.producer.check_source(host)
    }
}

// Length-only bootstrap; no graph traversal, gamma read or destination allocation.
// The earlier lookup remains separately charged. Controls cover simultaneous
// result/move slots and validation borrows; each word's init, validation and full
// cleanup is additional to the core quote. Core's 2048-byte logical-content
// envelope is reserved, NOT certified physical ABI/native-stack backing (HOLD).
fn execution_demand(shape: [usize;3]) -> Result<PrimitiveQuote,Error> {
    let [batch,rows,width]=shape;
    if batch==0 || rows==0 || width==0 { return Err(Error::Shape); }
    let rows=mul(batch,rows)?; let lanes=mul(rows,width)?;
    if rows>MAX_VALUES || lanes>MAX_VALUES { return Err(Error::Bounds); }
    let core=pllm_core::field_float::rms_norm_bf16_row_work(width).map_err(|_|Error::Bounds)?;
    let controls=[mul(3,size_of::<RawFirstNormActivation>())?,
        mul(2,size_of::<Result<RawFirstNormActivation,Error>>())?,
        mul(2,size_of::<std::sync::MutexGuard<'_,RawHostState>>())?,
        mul(32,size_of::<usize>())?,2048].into_iter().try_fold(0,plus)?;
    let work=[mul(rows,core)?,mul(lanes,64)?,mul(rows,128)?,controls,4096]
        .into_iter().try_fold(0,plus)?;
    PrimitiveQuote::checked(u64::try_from(work).map_err(|_|Error::Bounds)?,
        plus(mul(lanes,size_of::<u16>())?,controls)?)
}

// Revalidate concrete retained fields against the same immutable original plan;
// this never adopts a caller plan, gamma vector, file or alternate reader.
fn validate(norm: &AdmittedRawNormSpec) -> Result<(),Error> {
    let c=&norm.capability; let reservation=&norm.reservation;
    let producer=&reservation.producer; let input=&producer.source.input;
    let state=input.reader.root.state.lock().map_err(|_|Error::Binding)?;
    let payload=reader_payload(&state)?;
    let original=input.origin.upgrade().ok_or(Error::Binding)?;
    if !Arc::ptr_eq(&original,&input.reader.root) { return Err(Error::Binding); }
    check_header(&payload.header,&payload.directory,&payload.identity)?;
    if c.numeric_profile!=PROFILE || c.epsilon_bits!=0x358637bd || c.weight_offset!=0
        || c.shape!=input.spec.shape || norm.gamma.words.len()!=c.shape[2]
        || norm.gamma.words.capacity()!=c.shape[2] || producer.buffer.words.len()!=input.spec.lanes()?
        || producer.buffer.words.capacity()!=input.spec.lanes()?
        || c.lookup_operation!=input.spec.operation || reservation.operation.operation!=input.spec.operation
        || reservation.operation.tensor!=input.spec.tensor { return Err(Error::Binding); }
    if !matches!((c.graph,input.graph,reservation.operation.graph),
        (RawGraph::Prefill,RawGraph::Prefill,RawGraph::Prefill)|
        (RawGraph::Decode,RawGraph::Decode,RawGraph::Decode)) { return Err(Error::Binding); }
    let ops=graph_operations(payload,input.graph);
    let lookup=ops.get(input.spec.operation).ok_or(Error::Binding)?;
    if select_input_norm(&mut ops.iter().enumerate(),lookup,c.shape)?!=c.norm_operation { return Err(Error::Binding); }
    let op=ops.get(c.norm_operation).ok_or(Error::Binding)?;
    if norm_weight_offset(op.attributes.get("weight_offset"))?!=0
        || NormDecimal::new().parse(op.attributes.get("epsilon").and_then(|v|v.as_str()).ok_or(Error::Binding)?)?!=c.epsilon_bits {
        return Err(Error::Binding);
    }
    let gamma=payload.header.raw_premise.as_ref().ok_or(Error::Binding)?.tensors.get(c.gamma_tensor).ok_or(Error::Binding)?;
    if op.attributes.get("weight").and_then(|v|v.as_str())!=Some(gamma.name.as_str())
        || gamma.rank!=1 || gamma.shape[0]!=c.shape[2] || gamma.dtype!=raw::RawScalarType::Bf16Le
        || gamma.scale.is_some() || gamma.scale_index.is_some() || gamma.begin%2!=0
        || gamma.end.checked_sub(gamma.begin)!=Some(u64::try_from(mul(c.shape[2],2)?).map_err(|_|Error::Bounds)?) {
        return Err(Error::Binding);
    }
    let backing=payload.files.get(gamma.file).ok_or(Error::Binding)?;
    let file=backing.file.as_ref().ok_or(Error::Binding)?;
    let original=backing.original.upgrade().ok_or(Error::Binding)?;
    if !Arc::ptr_eq(&original,file) || !Arc::ptr_eq(file,&norm.gamma_file)
        || norm.extent[1]>file.size() || norm.extent[1].checked_sub(norm.extent[0])!=Some(u64::try_from(mul(c.shape[2],2)?).map_err(|_|Error::Bounds)?) {
        return Err(Error::Binding);
    }
    // Membership only verifies coverage; custody is the retained producer above.
    if !reservation.charge.is_for(payload.header.original.budget())
        || !input.charge.is_for(payload.header.original.budget())
        || !producer.buffer.charge.is_for(payload.header.original.budget()) { return Err(Error::Binding); }
    Ok(())
}

impl RawLookupActivation {
    /// Consumes this original attempt. Admission includes gamma ingestion, all
    /// rows, storage construction and failure cleanup BEFORE gamma/output factory.
    /// Work shortage burns siblings under the existing original-ledger contract;
    /// byte shortage retains its preceding debit. No partial output is published.
    pub(in crate::replicated_decoder) fn first_bf16_norm(self) -> Result<RawFirstNormActivation,Error> {
        let demand=execution_demand(self.source.input.spec.shape)?;
        let norm=self.admit_input_norm_spec_with(demand)?;
        execute(norm)
    }
}

// Private: the only production caller above supplies the combined reservation.
fn execute(norm: AdmittedRawNormSpec) -> Result<RawFirstNormActivation,Error> {
    #[cfg(test)]
    let mut norm=norm;
    #[cfg(test)]
    FAULT.with(|n|match n.get() {
        Some(0)=>norm.capability.numeric_profile="foreign",
        Some(1)=>norm.capability.graph=RawGraph::Decode,
        Some(2)=>norm.capability.gamma_tensor=usize::MAX,
        Some(3)=>norm.capability.epsilon_bits+=1,
        Some(4)=>norm.capability.shape[2]+=1,
        Some(5)=>norm.capability.norm_operation=usize::MAX,
        _=>{}
    });
    validate(&norm)?; // entirely covered by the combined semantic/admission quote
    let shape=norm.capability.shape;
    let lanes=mul(mul(shape[0],shape[1])?,shape[2])?;
    {
        let mut state=norm.reservation.producer.source.input.reader.root.state.lock().map_err(|_|Error::Binding)?;
        reader_payload(&state)?;
        state.reader_failed=true;
    } // never hold this mutex across source/output Drop on error
    #[cfg(test)]
    if let Some(row)=FAIL_ROW.with(|n|n.get()) {
        // Privileged test-only finite activation fault. The actual core square
        // overflows in this later row; no synthetic early return substitutes it.
        norm.reservation.producer.buffer.words[row*shape[2]]=0x7f7f;
    }
    let mut output=RawFirstNormActivation {words:exact_vec(lanes)?,norm};
    #[cfg(test)]
    OUTPUT_FACTORIES.with(|n|n.set(n.get()+1));
    output.words.resize(lanes,0);
    if output.words.capacity()!=lanes { return Err(Error::Bounds); }
    for (input,destination) in output.norm.reservation.producer.words().chunks_exact(shape[2])
        .zip(output.words.chunks_exact_mut(shape[2])) {
        pllm_core::field_float::rms_norm_bf16_row_into(input,&output.norm.gamma.words,destination)
            .map_err(|_|Error::Binding)?;
        #[cfg(test)]
        ROWS_DONE.with(|n|n.set(n.get()+1));
    }
    {
        let mut state=output.norm.reservation.producer.source.input.reader.root.state.lock().map_err(|_|Error::Binding)?;
        state.reader_failed=false;
    }
    Ok(output)
}

#[cfg(test)]
pub(in crate::replicated_decoder) fn quote_for_tests(output: &RawLookupActivation) -> PrimitiveQuote {
    let execution=execution_demand(output.shape()).unwrap();
    reservation_quote_for_tests(output,norm_spec_demand(output).unwrap().include(
        execution.work,execution.resident_bytes).unwrap())
}
#[cfg(test)]
pub(in crate::replicated_decoder) fn demand_for_tests(shape: [usize;3]) -> Result<PrimitiveQuote,Error> {
    execution_demand(shape)
}
#[cfg(test)]
impl RawFirstNormActivation {
    // Read-only test observation; cannot construct, mutate, rebind or authorize
    // a producer. Return only booleans, never copied source/inspection payloads.
    pub(in crate::replicated_decoder) fn original_gamma_matches_for_tests(&self, host: &RawHost,
        expected: &[u16]) -> bool {
        let norm=&self.norm; let c=&norm.capability;
        let input=&norm.reservation.producer.source.input;
        let state=input.reader.root.state.lock().unwrap();
        let payload=reader_payload(&state).unwrap();
        let ops=graph_operations(payload,input.graph);
        let op=&ops[c.norm_operation]; let lookup=&ops[c.lookup_operation];
        let gamma=&payload.header.raw_premise.as_ref().unwrap().tensors[c.gamma_tensor];
        let backing=&payload.files[gamma.file];
        Arc::ptr_eq(&input.reader.root,&host.root)
            && Arc::ptr_eq(&input.origin.upgrade().unwrap(),&host.root)
            && op.inputs.len()==1 && op.inputs[0]==lookup.id
            && op.attributes.get("weight").and_then(|v|v.as_str())==Some(gamma.name.as_str())
            && gamma.rank==1 && gamma.shape[0]==expected.len()
            && gamma.dtype==raw::RawScalarType::Bf16Le
            && norm.gamma.words.as_slice()==expected
            && norm.gamma.words.capacity()==expected.len()
            && Arc::ptr_eq(&norm.gamma_file,backing.file.as_ref().unwrap())
            && Arc::ptr_eq(&norm.gamma_file,&backing.original.upgrade().unwrap())
    }
}
#[cfg(test)]
thread_local! {
    pub(in crate::replicated_decoder) static OUTPUT_FACTORIES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)};
    pub(in crate::replicated_decoder) static OUTPUT_DROP:std::cell::Cell<Option<(bool,usize)>>=const {std::cell::Cell::new(None)};
    pub(in crate::replicated_decoder) static OUTPUT_PREWIPE:std::cell::Cell<Option<(usize,usize,bool)>>=const {std::cell::Cell::new(None)};
    pub(in crate::replicated_decoder) static ROWS_DONE:std::cell::Cell<usize>=const {std::cell::Cell::new(0)};
    pub(in crate::replicated_decoder) static FAIL_ROW:std::cell::Cell<Option<usize>>=const {std::cell::Cell::new(None)};
    pub(in crate::replicated_decoder) static FAULT:std::cell::Cell<Option<usize>>=const {std::cell::Cell::new(None)};
}
