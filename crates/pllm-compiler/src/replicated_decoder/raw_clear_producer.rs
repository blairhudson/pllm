//! Genuine TokenLookup and first BF16 RMSNorm prefix. No CLEAR projection, SDK activation, Q18
//! reinterpretation, caller-supplied BF16, independent budget or external lease.
//! This private module is a child of the original raw custody implementation.
use super::*;

#[path = "raw_first_norm.rs"]
pub(super) mod raw_first_norm;
pub(in crate::replicated_decoder) use raw_first_norm::RawFirstNormActivation;

#[derive(Clone, Copy, PartialEq, Eq)]
struct LookupSpec {
    operation: usize,
    tensor: usize,
    shape: [usize; 3],
    vocabulary: usize,
}
impl LookupSpec {
    fn count(self) -> Result<usize,Error> { mul(self.shape[0],self.shape[1]) }
    fn lanes(self) -> Result<usize,Error> { mul(self.count()?,self.shape[2]) }
}

/// Complete immutable original request, not an equality label. Reader ownership
/// is transferred once, not borrowed from a caller or reconstructed from pins.
pub(in crate::replicated_decoder) struct AdmittedRawInput {
    tokens: Vec<u64>,
    origin: std::sync::Weak<RootedRawHost>,
    graph: RawGraph,
    spec: LookupSpec,
    charge: OwnedExternalCharge,
    reader: RawReader, // liveness clears LAST, after tokens and charge destruction
}
impl Drop for AdmittedRawInput {
    fn drop(&mut self) {
        self.tokens.fill(0);
        #[cfg(test)]
        TOKEN_DROP.with(|s|s.set(Some((self.tokens.iter().all(|x|*x==0),self.charge.budget().used_bytes()))));
        drop(std::mem::take(&mut self.tokens)); // deallocate BEFORE charge release
    }
}

/// Newly created consuming-host seam. Not a pre-existing declaration wrapper.
pub(in crate::replicated_decoder) struct PrivateSourceInstance {
    input: AdmittedRawInput,
}

struct LookupBuffer {
    words: Vec<u16>,
    scratch: [u8; FRAME],
    workspace: SourceDecodeWorkspace,
    charge: OwnedExternalCharge, // covers ALL prior fields, including capacity
}
impl Drop for LookupBuffer {
    fn drop(&mut self) {
        self.words.resize(self.words.capacity(),0); // initialize unused tail without growing
        self.words.fill(0);
        self.scratch.fill(0);
        #[cfg(test)]
        LOOKUP_DROP.with(|s|s.set(Some((self.words.iter().all(|x|*x==0)
            && self.scratch.iter().all(|x|*x==0),self.charge.budget().used_bytes()))));
        drop(std::mem::take(&mut self.words));
        // SourceDecodeWorkspace's existing Drop wipes it before charge refund.
    }
}

/// Owned immutable BF16 matrix [batch, original query rows, original width].
/// Original ordered token IDs, operation, header, files, directory and ledger
/// remain owned through every borrow. No constructor accepts numeric buffers.
pub(in crate::replicated_decoder) struct RawLookupActivation {
    buffer: LookupBuffer,
    source: PrivateSourceInstance, // source/input/reader drop AFTER numeric storage
}
impl RawLookupActivation {
    pub(in crate::replicated_decoder) fn shape(&self) -> [usize;3] { self.source.input.spec.shape }
    /// Numeric view only: cannot certify a different request/source/producer.
    pub(in crate::replicated_decoder) fn words(&self) -> &[u16] { &self.buffer.words }

    /// Exact source identity, never matching contents, publisher pins or labels.
    /// This check grants no construction/publication authority to the argument.
    pub(in crate::replicated_decoder) fn check_source(&self, host: &RawHost) -> Result<(),Error> {
        let state=self.source.input.reader.root.state.lock().map_err(|_|Error::Binding)?;
        let payload=reader_payload(&state)?;
        let _control=admit_reader(payload,plus(size_of::<std::sync::MutexGuard<'_,RawHostState>>(),
            size_of::<OwnedExternalCharge>())?,64)?;
        if !Arc::ptr_eq(&self.source.input.reader.root,&host.root) { return Err(Error::Binding); }
        check_header(&payload.header,&payload.directory,&payload.identity)
    }

    #[cfg(test)]
    pub(super) fn tokens_for_tests(&self) -> &[u64] { &self.source.input.tokens }
    #[cfg(test)]
    pub(super) fn lookup_charge_for_tests(&self) -> (usize,usize) {
        (self.buffer.charge.payload_bytes(),self.buffer.charge.budget().used_bytes())
    }
}

fn graph_operations(payload: &RawRegistration, graph: RawGraph) -> &[pllm_models::ModelOperation] {
    let plan=payload.header.original.plan();
    match graph { RawGraph::Prefill=>&plan.prefill.operations, RawGraph::Decode=>&plan.decode.operations }
}

// Only scalar counts for the quote. No semantic string/shape traversal or token
// validation/copy until this complete original-counter admission succeeds.
fn input_quote(payload: &RawRegistration, graph: RawGraph, count: usize) -> Result<(usize,u64),Error> {
    let ops=graph_operations(payload,graph).len();
    let tensors=payload.header.raw_premise.as_ref().ok_or(Error::Binding)?.tensors.len();
    let bytes=plus(mul(count,size_of::<u64>())?,plus(
        mul(3,size_of::<AdmittedRawInput>())?,mul(2,size_of::<Result<AdmittedRawInput,Error>>())?)?)?;
    let work=[mul(count,16)?,mul(ops,4*NAME+64)?,mul(tensors,NAME+1)?,512]
        .into_iter().try_fold(0,plus)?;
    Ok((bytes,u64::try_from(work).map_err(|_|Error::Bounds)?))
}

fn lookup_spec(payload: &RawRegistration, graph: RawGraph) -> Result<LookupSpec,Error> {
    #[cfg(test)]
    LOOKUP_SPEC_VISITS.with(|n|n.set(n.get()+1));
    check_header(&payload.header,&payload.directory,&payload.identity)?;
    let ops=graph_operations(payload,graph);
    let mut selected=None;
    for (i,op) in ops.iter().enumerate() {
        // Model-neutral original graph input edge, not a caller operation name.
        if op.operator==pllm_models::ModelOperator::TokenLookup && op.inputs.len()==1
            && op.inputs[0]=="input.tokens" {
            if selected.replace(i).is_some() { return Err(Error::Binding); }
        }
    }
    let operation=selected.ok_or(Error::Binding)?;
    let op=&ops[operation];
    if op.output_shape.len()!=3 { return Err(Error::Shape); }
    let mut shape=[0usize;3];
    for (out,dimension) in shape.iter_mut().zip(&op.output_shape) {
        *out=usize::try_from(*dimension).map_err(|_|Error::Bounds)?;
        if *out==0 { return Err(Error::Shape); }
    }
    let name=original_weight(payload,graph,operation)?;
    let tensors=&payload.header.raw_premise.as_ref().ok_or(Error::Binding)?.tensors;
    let tensor=tensors.iter().position(|t|t.name==name).ok_or(Error::Binding)?;
    let weight=&tensors[tensor];
    if weight.rank!=2 || weight.dtype!=raw::RawScalarType::Bf16Le || weight.scale_index.is_some()
        || weight.scale.is_some() || weight.shape[0]==0 || weight.shape[1]!=shape[2]
        || weight.begin%2!=0 || weight.end.checked_sub(weight.begin)!=Some(
            u64::try_from(mul(weight.elements()?,2)?).map_err(|_|Error::Bounds)?) {
        return Err(Error::Shape);
    }
    let spec=LookupSpec {operation,tensor,shape,vocabulary:weight.shape[0]};
    if spec.count()?==0 || spec.count()?>MAX_VALUES || spec.lanes()?>MAX_VALUES { return Err(Error::Bounds); }
    Ok(spec)
}

impl RawReader {
    /// Caller IDs are REQUEST DATA only. Consumes this root's sole reader even
    /// on bounds/admission/late validation failure; never accepts caller BF16.
    pub(in crate::replicated_decoder) fn admit_tokens(self, graph: RawGraph, tokens: &[u64]) -> Result<AdmittedRawInput,Error> {
        let (charge,spec)={
            let mut state=self.root.state.lock().map_err(|_|Error::Binding)?;
            let payload=reader_payload(&state)?;
            let (bytes,work)=input_quote(payload,graph,tokens.len())?;
            state.reader_failed=true;
            let payload=state.payload.as_ref().ok_or(Error::Retired)?;
            let charge=admit_reader(payload,bytes,work)?;
            let spec=lookup_spec(payload,graph)?;
            if tokens.len()!=spec.count()? { return Err(Error::Shape); }
            // Bounded complete traversal only AFTER admission. No truncated row,
            // flat vector substitution, host usize truncation or modular IDs.
            for token in tokens {
                if usize::try_from(*token).map_err(|_|Error::Bounds)?>=spec.vocabulary { return Err(Error::Bounds); }
            }
            state.reader_failed=false;
            (charge,spec)
        };
        #[cfg(test)]
        TOKEN_FACTORIES.with(|n|n.set(n.get()+1));
        let origin=Arc::downgrade(&self.root);
        let mut input=AdmittedRawInput {tokens:exact_vec(tokens.len())?,origin,graph,spec,charge,reader:self};
        input.tokens.extend_from_slice(tokens);
        Ok(input)
    }
}

// Full output capacity plus simultaneous input, full scratch, scalar workspace,
// vector/result/move owners and transition controls. Input remains separately
// charged. No temporary destination is allocated before this original admission.
fn lookup_quote(payload: &RawRegistration, input: &AdmittedRawInput) -> Result<(usize,u64),Error> {
    let (semantic_bytes,semantic_work)=input_quote(payload,input.graph,0)?;
    let lanes=input.spec.lanes()?;
    let rows=input.spec.count()?;
    let row_bytes=mul(input.spec.shape[2],2)?;
    let windows=mul(rows,ceil(row_bytes,FRAME)?)?;
    let bytes=[mul(lanes,size_of::<u16>())?,semantic_bytes,
        mul(3,size_of::<LookupBuffer>())?,mul(2,size_of::<Result<RawLookupActivation,Error>>())?,
        size_of::<RawLookupActivation>(),mul(2,size_of::<SourceDecodeWorkspace>())?,
        mul(4,size_of::<Arc<PublicFile>>())?,size_of::<std::sync::MutexGuard<'_,RawHostState>>(),
        mul(16,size_of::<u64>())?,size_of::<[u8;8]>()].into_iter().try_fold(0,plus)?;
    let work=[usize::try_from(semantic_work).map_err(|_|Error::Bounds)?,
        mul(lanes,32)?,mul(rows,64)?,mul(windows,64)?,mul(8,FRAME)?,512]
        .into_iter().try_fold(0,plus)?;
    Ok((bytes,u64::try_from(work).map_err(|_|Error::Bounds)?))
}

impl PrivateSourceInstance {
    pub(in crate::replicated_decoder) fn new(input: AdmittedRawInput) -> Self { Self {input} }

    /// Consumes the original request attempt. Any late error destroys and wipes
    /// partial output, retains original host roots, and cannot replay this input.
    pub(in crate::replicated_decoder) fn token_lookup(self) -> Result<RawLookupActivation,Error> {
        let original=self.input.origin.upgrade().ok_or(Error::Binding)?;
        if !Arc::ptr_eq(&original,&self.input.reader.root) { return Err(Error::Binding); }
        let mut buffer={
            let mut state=self.input.reader.root.state.lock().map_err(|_|Error::Binding)?;
            let payload=reader_payload(&state)?;
            if !self.input.charge.is_for(payload.header.original.budget()) { return Err(Error::Binding); }
            let (bytes,work)=lookup_quote(payload,&self.input)?;
            state.reader_failed=true;
            let payload=state.payload.as_ref().ok_or(Error::Retired)?;
            let charge=admit_reader(payload,bytes,work)?;
            if lookup_spec(payload,self.input.graph)?!=self.input.spec { return Err(Error::Binding); }
            let spec=self.input.spec;
            if self.input.tokens.len()!=spec.count()? { return Err(Error::Binding); }
            let weight=&payload.header.raw_premise.as_ref().ok_or(Error::Binding)?.tensors[spec.tensor];
            let backing=payload.files.get(weight.file).ok_or(Error::Binding)?;
            let original=backing.original.upgrade().ok_or(Error::Binding)?;
            let file=backing.file.as_ref().ok_or(Error::Binding)?;
            if !Arc::ptr_eq(&original,file) { return Err(Error::Binding); }
            let start=data_start(file)?.checked_add(weight.begin).ok_or(Error::Bounds)?;
            let end=data_start(file)?.checked_add(weight.end).ok_or(Error::Bounds)?;
            if end>file.size() { return Err(Error::Bounds); }
            let row_bytes=mul(spec.shape[2],2)?;
            #[cfg(test)]
            LOOKUP_FACTORIES.with(|n|n.set(n.get()+1));
            let mut buffer=LookupBuffer {words:exact_vec(spec.lanes()?)?,scratch:[0;FRAME],
                workspace:SourceDecodeWorkspace::new(),charge};
            for (row,token) in self.input.tokens.iter().enumerate() {
                #[cfg(test)]
                if FAIL_ROW.with(|n|n.get())==Some(row) { return Err(Error::Binding); }
                let token=usize::try_from(*token).map_err(|_|Error::Bounds)?;
                if token>=spec.vocabulary { return Err(Error::Bounds); }
                let relative=mul(token,row_bytes)?;
                let begin=start.checked_add(u64::try_from(relative).map_err(|_|Error::Bounds)?).ok_or(Error::Bounds)?;
                let row_end=begin.checked_add(u64::try_from(row_bytes).map_err(|_|Error::Bounds)?).ok_or(Error::Bounds)?;
                // Alignment is relative to the original BF16 tensor, not the
                // filesystem address of a JSON header. No typed pointer cast:
                // unaligned file offsets are decoded from checked byte pairs.
                if relative%2!=0 || row_bytes%2!=0 || row_end>end { return Err(Error::Bounds); }
                let mut offset=0usize;
                while offset<row_bytes {
                    let len=(row_bytes-offset).min(FRAME);
                    if len%2!=0 { return Err(Error::Bounds); }
                    let absolute=begin.checked_add(u64::try_from(offset).map_err(|_|Error::Bounds)?).ok_or(Error::Bounds)?;
                    file.read(absolute,&mut buffer.scratch[..len]).map_err(|_|Error::Binding)?;
                    for bytes in buffer.scratch[..len].chunks_exact(2) {
                        raw::decode_finite(raw::RawScalarType::Bf16Le,bytes,&mut buffer.workspace).map_err(|_|Error::Binding)?;
                        // Exact original LEu16, including signed zero/subnormals.
                        // No Q18, Q8, F32 rounding, scaling or numeric adaptation.
                        buffer.words.push(u16::from_le_bytes([bytes[0],bytes[1]]));
                    }
                    buffer.scratch[..len].fill(0);
                    offset=plus(offset,len)?;
                }
            }
            if buffer.words.len()!=spec.lanes()? || buffer.words.capacity()!=spec.lanes()? { return Err(Error::Shape); }
            state.reader_failed=false;
            buffer
        };
        buffer.scratch.fill(0);
        Ok(RawLookupActivation {buffer,source:self})
    }
}

// Original-admission slice. No extraction, reattachment, Clone or borrowed-local
// lifetime: charge drops first, then the already independently charged producer.
// A future destination owner MUST wipe/free its full capacity before dropping
// this reservation. This slice grants ONLY original TokenLookup-associated work.
pub(super) struct OriginalReservation<P> {
    charge: OwnedExternalCharge,
    operation: OriginalOperationKey,
    producer: P,
}

fn reservation_demand<P>(demand: PrimitiveQuote) -> Result<PrimitiveQuote,Error> {
    let bytes=plus(mul(3,size_of::<OriginalReservation<P>>())?,
        mul(2,size_of::<Result<OriginalReservation<P>,Error>>())?)?;
    let work=u64::try_from(bytes).map_err(|_|Error::Bounds)?.checked_add(64).ok_or(Error::Bounds)?;
    demand.include(work,bytes)
}

impl RawLookupActivation {
    pub(super) fn reserve_original(self, demand: PrimitiveQuote) -> Result<OriginalReservation<Self>,Error> {
        let input=&self.source.input;
        let key=OriginalOperationKey {graph:input.graph,operation:input.spec.operation,tensor:input.spec.tensor};
        self.reserve_original_key(key,demand)
    }

    fn reserve_original_key(self, key: OriginalOperationKey, demand: PrimitiveQuote) -> Result<OriginalReservation<Self>,Error> {
        let demand=reservation_demand::<Self>(demand)?;
        let reader=&self.source.input.reader;
        #[cfg(not(test))]
        let charge=admit_primitive(reader,&key,demand)?;
        #[cfg(test)]
        let charge=admit_primitive(reader,&key,demand).map_err(|error| {
            ORIGINAL_REFUSAL.with(|n|n.set(Some((self.buffer.words.len(),self.buffer.words.capacity(),
                self.buffer.charge.payload_bytes(),self.buffer.charge.budget().used_bytes()))));
            error
        })?;
        {
            let mut state=reader.root.state.lock().map_err(|_|Error::Binding)?;
            reader_payload(&state)?;
            state.reader_failed=true;
            let payload=state.payload.as_ref().ok_or(Error::Retired)?;
            let input=&self.source.input;
            let original=input.origin.upgrade().ok_or(Error::Binding)?;
            if !Arc::ptr_eq(&original,&reader.root) || !input.charge.is_for(payload.header.original.budget())
                || !charge.is_for(payload.header.original.budget()) { return Err(Error::Binding); }
            // All original semantic work is covered by admit_primitive BEFORE
            // this traversal. An admitted demand is NOT a different op license.
            let same_graph=matches!((key.graph,input.graph),
                (RawGraph::Prefill,RawGraph::Prefill)|(RawGraph::Decode,RawGraph::Decode));
            if !same_graph { return Err(Error::Binding); }
            let spec=lookup_spec(payload,input.graph)?;
            if key.operation!=spec.operation || key.tensor!=spec.tensor || spec!=input.spec
                || input.tokens.len()!=spec.count()? || self.buffer.words.len()!=spec.lanes()?
                || self.buffer.words.capacity()!=spec.lanes()? { return Err(Error::Binding); }
            state.reader_failed=false;
        }
        Ok(OriginalReservation {charge,operation:key,producer:self})
    }
}

#[cfg(test)]
pub(super) fn reservation_quote_for_tests(output: &RawLookupActivation, demand: PrimitiveQuote) -> PrimitiveQuote {
    let input=&output.source.input;
    let state=input.reader.root.state.lock().unwrap();
    let key=OriginalOperationKey {graph:input.graph,operation:input.spec.operation,tensor:input.spec.tensor};
    primitive_admission_quote(state.payload.as_ref().unwrap(),&key,reservation_demand::<RawLookupActivation>(demand).unwrap()).unwrap()
}

#[cfg(test)]
pub(super) fn reserve_forged_key_for_tests(output: RawLookupActivation, demand: PrimitiveQuote,
    graph: RawGraph, operation: usize, tensor: usize) -> Result<OriginalReservation<RawLookupActivation>,Error> {
    output.reserve_original_key(OriginalOperationKey {graph,operation,tensor},demand)
}
#[cfg(test)]
pub(super) fn replace_activation_reader_for_tests(output: &mut RawLookupActivation, reader: RawReader) {
    output.source.input.reader=reader;
}

// Norm SPEC/gamma admission only: no norm output, rsqrt, fanout or CLEAR.
const NORM_NODES: usize = 1<<20;
struct NormCapability {
    graph: RawGraph,
    lookup_operation: usize,
    norm_operation: usize,
    gamma_tensor: usize,
    shape: [usize;3],
    weight_offset: u64,
    epsilon_bits: u32,
    numeric_profile: &'static str,
}
struct NormGamma {
    words: Vec<u16>,
    #[cfg(test)]
    header: Arc<SourceHeaderEntry>,
}
impl Drop for NormGamma {
    fn drop(&mut self) {
        #[cfg(test)]
        NORM_GAMMA_PREWIPE.with(|n|n.set(Some((self.words.len(),self.words.capacity()))));
        self.words.resize(self.words.capacity(),0);
        self.words.fill(0);
        #[cfg(test)]
        let wiped=self.words.iter().all(|w|*w==0);
        drop(std::mem::take(&mut self.words));
        #[cfg(test)]
        NORM_GAMMA_DROP.with(|n|n.set(Some((wiped && self.words.capacity()==0,self.header.root.budget().used_bytes()))));
    }
}
pub(super) struct AdmittedRawNormSpec {
    gamma: NormGamma, // full capacity wiped/freed BEFORE covering reservation
    capability: NormCapability,
    gamma_file: Arc<PublicFile>,
    extent: [u64;2],
    reservation: OriginalReservation<RawLookupActivation>,
}

// Allocation-free one-pass decimal state. Only one nonzero significand digit,
// exactly '1'; E-F+Z == -6 proves equality, not merely the same rounded F32.
struct NormDecimal {
    index: usize, fractional: i64, zeros: i64, exponent: i64,
    phase: u8, dot: bool, digit: bool, one: bool, exponent_digit: bool, negative_exponent: bool,
}
impl NormDecimal {
    fn new() -> Self { Self {index:0,fractional:0,zeros:0,exponent:0,phase:0,dot:false,
        digit:false,one:false,exponent_digit:false,negative_exponent:false} }
    fn parse(&mut self, text: &str) -> Result<u32,Error> {
        if text.is_empty() || text.len()>NAME { return Err(Error::Binding); }
        while self.index<text.len() {
            let b=text.as_bytes()[self.index];
            let first=self.index==0; self.index+=1;
            if first && b==b'+' { continue; }
            if self.phase==0 {
                match b {
                    b'0'..=b'9'=> {
                        self.digit=true;
                        if self.dot { self.fractional=self.fractional.checked_add(1).ok_or(Error::Bounds)?; }
                        if b==b'1' && !self.one { self.one=true; }
                        else if b==b'0' { if self.one { self.zeros=self.zeros.checked_add(1).ok_or(Error::Bounds)?; } }
                        else { return Err(Error::Binding); }
                    }
                    b'.' if !self.dot=>self.dot=true,
                    b'e'|b'E' if self.digit && self.one=>self.phase=1,
                    _=>return Err(Error::Binding),
                }
            } else {
                if self.phase==1 && (b==b'+' || b==b'-') {
                    self.negative_exponent=b==b'-'; self.phase=2; continue;
                }
                if !b.is_ascii_digit() { return Err(Error::Binding); }
                self.phase=2; self.exponent_digit=true;
                self.exponent=self.exponent.checked_mul(10).and_then(|e|e.checked_add(i64::from(b-b'0'))).ok_or(Error::Bounds)?;
                if self.exponent>i64::try_from(plus(mul(2,NAME)?,6)?).map_err(|_|Error::Bounds)? { return Err(Error::Bounds); }
            }
        }
        if !self.digit || !self.one || (self.phase!=0 && !self.exponent_digit) { return Err(Error::Binding); }
        let exponent=if self.negative_exponent {-self.exponent} else {self.exponent};
        if exponent.checked_sub(self.fractional).and_then(|e|e.checked_add(self.zeros))!=Some(-6) { return Err(Error::Binding); }
        Ok(0x358637bd)
    }
}

// Actual simultaneous local aggregate. Guard borrows the external reservation;
// semantic borrows/iterators borrow a separate retained header Arc, NOT ourselves.
// Its concrete layout quotes parser, borrowed attrs, ordinals, capability staging,
// header I/O, temporary identity Arcs/Weak, quote, cleanup and all frame storage.
struct NormAdmissionLocals<'a> {
    guard: std::sync::MutexGuard<'a,RawHostState>,
    operations: std::iter::Enumerate<std::slice::Iter<'a,pllm_models::ModelOperation>>,
    tensors: std::iter::Enumerate<std::slice::Iter<'a,ExpectedTensor>>,
    attributes: Option<serde_json::map::Iter<'a>>,
    lookup: &'a pllm_models::ModelOperation,
    selected: Option<(usize,&'a pllm_models::ModelOperation)>,
    weight: Option<&'a serde_json::Value>, offset: Option<&'a serde_json::Value>, epsilon: Option<&'a serde_json::Value>,
    parser: NormDecimal,
    tensor: Option<usize>, shape: [usize;3], extent: [u64;2], epsilon_bits: u32,
    bytes: usize, read_offset: usize, window: usize, header_bytes: [u8;8],
    quote: PrimitiveQuote, capability: Option<NormCapability>,
    coverage: &'a OwnedExternalCharge,
    scratch: [u8;FRAME], workspace: SourceDecodeWorkspace,
    weak: Option<std::sync::Weak<PublicFile>>, original: Option<Arc<PublicFile>>,
    gamma_file: Option<Arc<PublicFile>>, // file remains through scratch/workspace cleanup
}
impl Drop for NormAdmissionLocals<'_> {
    fn drop(&mut self) {
        self.scratch.fill(0); self.header_bytes.fill(0);
        #[cfg(test)]
        NORM_SCRATCH_DROP.with(|n|n.set(Some((self.scratch.iter().all(|b|*b==0),self.coverage.budget().used_bytes()))));
        // workspace's existing Drop precedes temporary and retained file Arcs.
    }
}

fn norm_spec_demand(output: &RawLookupActivation) -> Result<PrimitiveQuote,Error> {
    let input=&output.source.input;
    let state=input.reader.root.state.lock().map_err(|_|Error::Binding)?;
    let payload=reader_payload(&state)?;
    let n=graph_operations(payload,input.graph).len();
    let t=payload.header.raw_premise.as_ref().ok_or(Error::Binding)?.tensors.len();
    let d=input.spec.shape[2];
    if n>NORM_NODES || t>source_policy::DESCRIPTORS || d==0 { return Err(Error::Bounds); }
    let g=mul(2,d)?; let w=ceil(g,FRAME)?; let b=size_of::<SourceDecodeWorkspace>(); let s=mul(16,NAME+1)?;
    let c=[mul(3,size_of::<AdmittedRawNormSpec>())?,mul(2,size_of::<Result<AdmittedRawNormSpec,Error>>())?,
        mul(3,size_of::<NormAdmissionLocals<'_>>())?].into_iter().try_fold(0,plus)?;
    let work=[4096,mul(n,s)?,mul(NORM_NODES,128)?,mul(128,NAME+1)?,mul(t,s)?,
        mul(w,128)?,mul(g,16)?,mul(d,128)?,mul(16,plus(plus(g,FRAME)?,b)?)?,c]
        .into_iter().try_fold(0,plus)?;
    PrimitiveQuote::checked(u64::try_from(work).map_err(|_|Error::Bounds)?,
        [g,FRAME,b,c].into_iter().try_fold(0,plus)?)
}

// Scalar selector only, not a norm capability or source constructor. Production
// calls it once with the immutable original operations; component tests cannot
// mint AdmittedRawNormSpec by supplying an unrelated graph.
fn select_input_norm(operations: &mut std::iter::Enumerate<std::slice::Iter<'_,pllm_models::ModelOperation>>,
    lookup: &pllm_models::ModelOperation, shape: [usize;3]) -> Result<usize,Error> {
    if lookup.id.len()>NAME { return Err(Error::Bounds); }
    let mut selected=None;
    for (index,op) in operations {
        if op.operator!=pllm_models::ModelOperator::RmsNorm || op.inputs.len()!=1 { continue; }
        if op.inputs[0].len()>NAME { return Err(Error::Bounds); }
        if op.inputs[0]!=lookup.id { continue; }
        if selected.replace(index).is_some() { return Err(Error::Binding); }
        if op.output_shape.len()!=3 { return Err(Error::Shape); }
        for axis in 0..3 { if usize::try_from(op.output_shape[axis]).map_err(|_|Error::Bounds)?!=shape[axis] { return Err(Error::Shape); } }
    }
    selected.ok_or(Error::Binding)
}
fn norm_weight_offset(value: Option<&serde_json::Value>) -> Result<u64,Error> {
    match value.and_then(|v|v.as_u64()) {Some(0)=>Ok(0),_=>Err(Error::Binding)}
}

impl RawLookupActivation {
    pub(super) fn admit_input_norm_spec(self) -> Result<AdmittedRawNormSpec,Error> {
        self.admit_input_norm_spec_with(PrimitiveQuote::checked(0,0)?)
    }

    // Additional concrete descendant storage/work is admitted TOGETHER with
    // gamma reads, before either factory. Reservation still owns this producer.
    fn admit_input_norm_spec_with(self, extra: PrimitiveQuote) -> Result<AdmittedRawNormSpec,Error> {
        let demand=norm_spec_demand(&self)?.include(extra.work,extra.resident_bytes)?;
        let reservation=self.reserve_original(demand)?; // key stays TokenLookup
        let input=&reservation.producer.source.input;
        let mut guard=input.reader.root.state.lock().map_err(|_|Error::Binding)?;
        reader_payload(&guard)?; guard.reader_failed=true;
        let payload=guard.payload.as_ref().ok_or(Error::Retired)?;
        check_header(&payload.header,&payload.directory,&payload.identity)?;
        let header=Arc::clone(&payload.header.header);
        let plan=header.original.plan();
        let ops=match input.graph {RawGraph::Prefill=>&plan.prefill.operations,RawGraph::Decode=>&plan.decode.operations};
        let premise=header.raw_premise.as_ref().ok_or(Error::Binding)?;
        if ops.len()>NORM_NODES || premise.tensors.len()>source_policy::DESCRIPTORS { return Err(Error::Bounds); }
        let lookup=ops.get(input.spec.operation).ok_or(Error::Binding)?;
        let mut locals=NormAdmissionLocals {guard,operations:ops.iter().enumerate(),tensors:premise.tensors.iter().enumerate(),
            attributes:None,lookup,selected:None,weight:None,offset:None,epsilon:None,parser:NormDecimal::new(),tensor:None,
            shape:input.spec.shape,extent:[0;2],epsilon_bits:0,bytes:mul(2,input.spec.shape[2])?,read_offset:0,window:0,
            header_bytes:[0;8],quote:demand,capability:None,coverage:&reservation.charge,scratch:[0;FRAME],
            workspace:SourceDecodeWorkspace::new(),weak:None,original:None,gamma_file:None};
        let selected=select_input_norm(&mut locals.operations,locals.lookup,locals.shape)?;
        locals.selected=Some((selected,&ops[selected]));
        let attrs=locals.selected.as_ref().ok_or(Error::Binding)?.1.attributes.as_object().ok_or(Error::Binding)?;
        if attrs.len()>NORM_NODES { return Err(Error::Bounds); }
        locals.attributes=Some(attrs.iter());
        for (key,value) in locals.attributes.as_mut().ok_or(Error::Binding)? {
            if key.len()>NAME { return Err(Error::Bounds); }
            match key.as_str() {"weight"=>locals.weight=Some(value),"weight_offset"=>locals.offset=Some(value),
                "epsilon"=>locals.epsilon=Some(value),_=>{}}
        }
        let weight=locals.weight.and_then(|v|v.as_str()).ok_or(Error::Binding)?;
        if weight.is_empty() || weight.len()>NAME { return Err(Error::Binding); }
        let weight_offset=norm_weight_offset(locals.offset)?;
        locals.epsilon_bits=locals.parser.parse(locals.epsilon.and_then(|v|v.as_str()).ok_or(Error::Binding)?)?;
        for (index,tensor) in &mut locals.tensors {
            if tensor.name.len()>NAME { return Err(Error::Bounds); }
            if tensor.name==weight && locals.tensor.replace(index).is_some() { return Err(Error::Binding); }
        }
        let tensor=locals.tensor.ok_or(Error::Binding)?;
        let gamma=&premise.tensors[tensor];
        if gamma.rank!=1 || gamma.shape[0]!=locals.shape[2] || gamma.dtype!=raw::RawScalarType::Bf16Le
            || gamma.scale.is_some() || gamma.scale_index.is_some() || gamma.begin%2!=0
            || gamma.end.checked_sub(gamma.begin)!=Some(u64::try_from(locals.bytes).map_err(|_|Error::Bounds)?) { return Err(Error::Shape); }
        let payload=locals.guard.payload.as_ref().ok_or(Error::Retired)?;
        let backing=payload.files.get(gamma.file).ok_or(Error::Binding)?;
        locals.weak=Some(backing.original.clone());
        locals.original=locals.weak.as_ref().and_then(|w|w.upgrade());
        let file=backing.file.as_ref().ok_or(Error::Binding)?;
        if !Arc::ptr_eq(locals.original.as_ref().ok_or(Error::Binding)?,file) { return Err(Error::Binding); }
        locals.gamma_file=Some(Arc::clone(file));
        #[cfg(test)]
        NORM_READS.with(|n|n.set(n.get()+1));
        let start=data_start_into(file,&mut locals.header_bytes)?; // ONE original header read
        locals.extent=[start.checked_add(gamma.begin).ok_or(Error::Bounds)?,start.checked_add(gamma.end).ok_or(Error::Bounds)?];
        if locals.extent[0]>=locals.extent[1] || locals.extent[1]>file.size() { return Err(Error::Bounds); }
        #[cfg(test)]
        NORM_FACTORIES.with(|n|n.set(n.get()+1));
        let mut words=NormGamma {words:exact_vec(locals.shape[2])?,#[cfg(test)] header:Arc::clone(&header)};
        while locals.read_offset<locals.bytes {
            let len=(locals.bytes-locals.read_offset).min(FRAME);
            if len%2!=0 { return Err(Error::Bounds); }
            let absolute=locals.extent[0].checked_add(u64::try_from(locals.read_offset).map_err(|_|Error::Bounds)?).ok_or(Error::Bounds)?;
            if absolute.checked_add(u64::try_from(len).map_err(|_|Error::Bounds)?).ok_or(Error::Bounds)?>locals.extent[1] { return Err(Error::Bounds); }
            #[cfg(test)]
            NORM_READS.with(|n|n.set(n.get()+1));
            file.read(absolute,&mut locals.scratch[..len]).map_err(|_|Error::Binding)?;
            #[cfg(test)]
            if NORM_FAIL_READ.with(|n|n.get())==Some(locals.window) {
                NORM_READ_FAULT_FIRED.with(|n|n.set(Some(locals.window)));
                return Err(Error::Binding);
            }
            for bytes in locals.scratch[..len].chunks_exact_mut(2) {
                #[cfg(test)]
                if NORM_NONFINITE_AT.with(|n|n.get())==Some(words.words.len()) { bytes.copy_from_slice(&0x7f80u16.to_le_bytes()); }
                raw::decode_finite(raw::RawScalarType::Bf16Le,bytes,&mut locals.workspace).map_err(|_|Error::Binding)?;
                words.words.push(u16::from_le_bytes([bytes[0],bytes[1]]));
            }
            locals.scratch[..len].fill(0);
            locals.read_offset=plus(locals.read_offset,len)?; locals.window=plus(locals.window,1)?;
        }
        if words.words.len()!=locals.shape[2] || words.words.capacity()!=locals.shape[2]
            || locals.quote.resident_bytes<locals.bytes { return Err(Error::Bounds); }
        locals.capability=Some(NormCapability {graph:input.graph,lookup_operation:input.spec.operation,norm_operation:selected,
            gamma_tensor:tensor,shape:locals.shape,weight_offset,epsilon_bits:locals.epsilon_bits,
            numeric_profile:raw_first_norm::PROFILE});
        let capability=locals.capability.take().ok_or(Error::Binding)?;
        let gamma_file=locals.gamma_file.take().ok_or(Error::Binding)?; let extent=locals.extent;
        locals.guard.reader_failed=false;
        drop(locals); drop(header);
        Ok(AdmittedRawNormSpec {gamma:words,capability,gamma_file,extent,reservation})
    }
}

#[cfg(test)]
pub(super) fn norm_quote_for_tests(output: &RawLookupActivation) -> PrimitiveQuote {
    reservation_quote_for_tests(output,norm_spec_demand(output).unwrap())
}
#[cfg(test)]
pub(super) fn norm_epsilon_for_tests(text: &str) -> Result<u32,Error> { NormDecimal::new().parse(text) }
#[cfg(test)]
pub(super) fn norm_offset_for_tests(value: Option<&serde_json::Value>) -> Result<u64,Error> { norm_weight_offset(value) }
#[cfg(test)]
pub(super) fn norm_operation_for_tests(operations: &[pllm_models::ModelOperation], lookup: usize, shape: [usize;3]) -> Result<usize,Error> {
    select_input_norm(&mut operations.iter().enumerate(),operations.get(lookup).ok_or(Error::Binding)?,shape)
}
#[cfg(test)]
impl AdmittedRawNormSpec {
    pub(super) fn snapshot_for_tests(&self) -> ([usize;3],u32,usize,usize,usize,u64,[u64;2],&[u16]) {
        let c=&self.capability; let key=&self.reservation.operation;
        assert!(matches!((key.graph,c.graph),(RawGraph::Prefill,RawGraph::Prefill)|(RawGraph::Decode,RawGraph::Decode)));
        assert_eq!(key.operation,c.lookup_operation);
        assert_eq!(self.gamma.words.len(),self.gamma.words.capacity());
        assert!(self.extent[1]<=self.gamma_file.size());
        (c.shape,c.epsilon_bits,c.lookup_operation,c.norm_operation,c.gamma_tensor,c.weight_offset,self.extent,&self.gamma.words)
    }
}

// Concrete test destination ONLY; proves owner/drop order without inventing a
// future norm/projection operation or using a caller callback as authority.
#[cfg(test)]
impl OriginalReservation<RawLookupActivation> {
    pub(super) fn tokens_for_tests(&self) -> &[u64] { &self.producer.source.input.tokens }
}
#[cfg(test)]
pub(super) struct ReservedTestBuffer {
    bytes: Vec<u8>,
    reservation: OriginalReservation<RawLookupActivation>,
}
#[cfg(test)]
impl Drop for ReservedTestBuffer {
    fn drop(&mut self) {
        let covered=self.reservation.charge.budget().used_bytes();
        self.bytes.resize(self.bytes.capacity(),0);
        self.bytes.fill(0);
        let wiped=self.bytes.iter().all(|b|*b==0);
        drop(std::mem::take(&mut self.bytes));
        RESERVED_DROP.with(|s|s.set(Some((wiped && self.bytes.capacity()==0
            && self.reservation.charge.budget().used_bytes()==covered,covered))));
        // Automatic reservation Drop refunds NEW demand, then destroys the old
        // independently charged BF16 producer/input. Original reader drops last.
    }
}
#[cfg(test)]
pub(super) fn reserved_test_demand() -> PrimitiveQuote {
    let controls=plus(mul(3,size_of::<ReservedTestBuffer>()).unwrap(),
        mul(2,size_of::<Result<ReservedTestBuffer,Error>>()).unwrap()).unwrap();
    PrimitiveQuote::checked(u64::try_from(controls).unwrap().checked_add(1024).unwrap(),plus(32,controls).unwrap()).unwrap()
}
#[cfg(test)]
pub(super) fn reserved_test_factory(output: RawLookupActivation, fail: bool) -> Result<ReservedTestBuffer,Error> {
    let reservation=output.reserve_original(reserved_test_demand())?;
    RESERVED_FACTORIES.with(|n|n.set(n.get()+1));
    let mut result=ReservedTestBuffer {bytes:exact_vec(32)?,reservation};
    result.bytes.resize(32,0);
    result.bytes[..17].fill(0x5a);
    if fail { return Err(Error::Binding); }
    Ok(result)
}
#[cfg(test)]
impl ReservedTestBuffer {
    pub(super) fn retained_for_tests(&self) -> ([usize;3],&[u16],usize) {
        assert_eq!(self.reservation.operation.operation,self.reservation.producer.source.input.spec.operation);
        assert_eq!(self.reservation.operation.tensor,self.reservation.producer.source.input.spec.tensor);
        (self.reservation.producer.shape(),self.reservation.producer.words(),self.bytes.capacity())
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static ORIGINAL_REFUSAL:std::cell::Cell<Option<(usize,usize,usize,usize)>>=const {std::cell::Cell::new(None)};
    pub(super) static NORM_GAMMA_PREWIPE:std::cell::Cell<Option<(usize,usize)>>=const {std::cell::Cell::new(None)};
    pub(super) static NORM_READ_FAULT_FIRED:std::cell::Cell<Option<usize>>=const {std::cell::Cell::new(None)};
    pub(super) static NORM_FACTORIES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)};
    pub(super) static NORM_READS:std::cell::Cell<usize>=const {std::cell::Cell::new(0)};
    pub(super) static NORM_FAIL_READ:std::cell::Cell<Option<usize>>=const {std::cell::Cell::new(None)};
    pub(super) static NORM_NONFINITE_AT:std::cell::Cell<Option<usize>>=const {std::cell::Cell::new(None)};
    pub(super) static NORM_GAMMA_DROP:std::cell::Cell<Option<(bool,usize)>>=const {std::cell::Cell::new(None)};
    pub(super) static NORM_SCRATCH_DROP:std::cell::Cell<Option<(bool,usize)>>=const {std::cell::Cell::new(None)};
    pub(super) static LOOKUP_SPEC_VISITS:std::cell::Cell<usize>=const {std::cell::Cell::new(0)};
    pub(super) static RESERVED_FACTORIES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)};
    pub(super) static RESERVED_DROP:std::cell::Cell<Option<(bool,usize)>>=const {std::cell::Cell::new(None)};
    pub(super) static TOKEN_FACTORIES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)};
    pub(super) static LOOKUP_FACTORIES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)};
    pub(super) static FAIL_ROW:std::cell::Cell<Option<usize>>=const {std::cell::Cell::new(None)};
    pub(super) static LOOKUP_DROP:std::cell::Cell<Option<(bool,usize)>>=const {std::cell::Cell::new(None)};
    pub(super) static TOKEN_DROP:std::cell::Cell<Option<(bool,usize)>>=const {std::cell::Cell::new(None)};
}

#[cfg(test)]
pub(super) fn input_quote_for_tests(host: &RawHost, graph: RawGraph, count: usize) -> (usize,u64) {
    let state=host.root.state.lock().unwrap(); input_quote(state.payload.as_ref().unwrap(),graph,count).unwrap()
}
#[cfg(test)]
pub(super) fn lookup_quote_for_tests(input: &AdmittedRawInput) -> (usize,u64) {
    let state=input.reader.root.state.lock().unwrap(); lookup_quote(state.payload.as_ref().unwrap(),input).unwrap()
}

#[cfg(test)]
pub(super) fn replace_operation_for_tests(input: &mut AdmittedRawInput, operation: usize) {
    input.spec.operation=operation;
}
#[cfg(test)]
pub(super) fn replace_reader_for_tests(input: &mut AdmittedRawInput, reader: RawReader) {
    input.reader=reader;
}
