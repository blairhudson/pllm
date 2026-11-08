//! Borrowed, consuming public forward-factory admission. Logical metadata only;
//! numerical bodies are deliberately separate, and no protected funding/custody
//! or decoder quote is adopted by this path.
use crate::{ConfigLowererLimits, ModelError, ForwardPlan, ForwardOperation, ForwardRequirement,
    ForwardDimension, ForwardOutput, ForwardTensorSpec};
use crate::encoder_blocks::{Geometry, OutputShape, BLOCK_OUTPUTS, EMBEDDING_OUTPUTS, POOLER_OUTPUTS};
use crate::forward_plan::invalid;
use crate::source_mappings::EncoderSource;
use crate::source_work::{self, SourceInputLimits, SOURCE_FRAME_SLOTS};
use pllm_types::bounded_canonical::borrowed::{NativeFrame, Traversal, Event};
use serde_json::Value;
use std::mem::size_of;

#[derive(Clone, Copy, Debug)]
pub struct ForwardLimits {
    pub source_bytes: usize,
    pub source: SourceInputLimits,
    pub metadata: ConfigLowererLimits,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForwardNumericalSummary {
    pub parameter_elements: u64,
    pub operation_output_elements: u64,
    pub largest_operation_elements: u64,
    pub input_bytes: u64,
    pub largest_input_bytes: u64,
    pub public_output_elements: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Conservative simultaneous logical metadata reservation for ONE factory
/// lifecycle, deterministic for the identical borrowed source (including its
/// existing capacities) and sequence, independent of admitted caller caps.
///
/// `metadata_peak_bytes` ALREADY includes original Value backing, source-digest
/// staging, retained plan including its ORIGINAL semantic-config copy, one concurrent
/// reconstruction plan, model-owned canonical tree/output/sort/scratch buffers,
/// one owned requirement Vec and its indexes/scratch. Sequential validation
/// reuses this reservation; additional concurrent reconstructions do not.
///
/// `source_backing_bytes`, `retained_plan_bytes`, `retained_requirements_bytes`
/// overlap that peak: observational subtotals, NEVER disjoint additions to it.
/// Excluded: original raw byte buffers, frontend snapshots/composition owners,
/// native-owned retained canonical/schedule/serialized-requirement buffers,
/// compiler step/IR storage, numerical FP32/input/output bodies, native compiled
/// call stacks, allocator overhead and RSS. Such independent owners are extras;
/// a native copy of a model buffer is not that model buffer counted twice.
/// `metadata_work` covers bounded Value census, scalar/label validation,
/// construction/reconstruction, model canonicalization and requirement loops;
/// byte-parser work is independently metered before Value allocation.
pub struct ForwardResourceQuote {
    /// Number of ordered model operations; not a construction permit.
    pub operation_count: usize,
    /// Exact aggregate UTF8 ID lengths, excludes String headers/capacity and Vec
    /// storage. Native may reserve its own exact copied-ID storage from this.
    pub operation_id_bytes: usize,
    pub source_backing_bytes: usize,
    pub retained_plan_bytes: usize,
    pub retained_requirements_bytes: usize,
    pub metadata_peak_bytes: usize,
    pub metadata_work: u64,
    pub numerical: ForwardNumericalSummary,
}

/// Immutable source/sequence binding. The numeric summary is descriptive; native
/// callers must reject impossible numerical allocations BEFORE consuming this.
///
/// ```compile_fail
/// use pllm_models::{preflight_forward_value,default_forward_limits};
/// let source=serde_json::json!({});
/// let p=preflight_forward_value(&source,1,default_forward_limits().unwrap()).unwrap();
/// let _first=p.construct();
/// let _second=p.construct();
/// ```
/// ```compile_fail
/// use pllm_models::{ForwardPreflight,preflight_forward_value,default_forward_limits};
/// fn escape<'a>() -> ForwardPreflight<'a> {
///   let source=serde_json::json!({});
///   preflight_forward_value(&source,1,default_forward_limits().unwrap()).unwrap()
/// }
/// ```
/// ```compile_fail
/// use pllm_models::{preflight_forward_value,default_forward_limits};
/// let source=serde_json::json!({});
/// let p=preflight_forward_value(&source,1,default_forward_limits().unwrap()).unwrap();
/// let _copy=p.clone();
/// ```
/// ```compile_fail
/// let _=serde_json::from_str::<pllm_models::ForwardPreflight<'_>>("{}");
/// ```
pub struct ForwardPreflight<'a> {
    source: &'a Value,
    sequence: u64,
    geometry: Geometry<'a>,
    registry: &'static EncoderSource,
    limits: ForwardLimits,
    quote: ForwardResourceQuote,
}
impl ForwardPreflight<'_> {
    pub fn quote(&self) -> &ForwardResourceQuote { &self.quote }
    pub fn construct(self) -> Result<ForwardPlan,ModelError> {
        // First source copy, decimal String, graph Vec, artifact name and digest
        // are all downstream of caller admission. The source witness/digest use
        // the ORIGINAL admitted Value (including exact decimal text); only the
        // builder's numeric operator attributes normalize to source FP32.
        let plan = crate::encoder_blocks::lower(self.source,self.sequence,self.geometry,self.registry)?;
        let _ = self.limits; // retained immutable admission policy, never rebound
        Ok(plan)
    }
}

fn overflow() -> ModelError { invalid("forward resource arithmetic overflow") }
pub(super) fn add(a: usize,b: usize) -> Result<usize,ModelError> { a.checked_add(b).ok_or_else(overflow) }
pub(super) fn mul(a: usize,b: usize) -> Result<usize,ModelError> { a.checked_mul(b).ok_or_else(overflow) }
fn sum(v: &[usize]) -> Result<usize,ModelError> { v.iter().try_fold(0,|n,v| add(n,*v)) }
fn u(n: usize) -> Result<u64,ModelError> { u64::try_from(n).map_err(|_| overflow()) }
fn a(a:u64,b:u64)->Result<u64,ModelError>{ a.checked_add(b).ok_or_else(overflow) }
fn m(a:u64,b:u64)->Result<u64,ModelError>{ a.checked_mul(b).ok_or_else(overflow) }
fn tree<T>() -> Result<usize,ModelError> {
    crate::owned_backing::checked_tree_node_bytes::<T>().map_err(|_| overflow())
}
fn digits(mut n:u64)->usize { let mut digits=1; while n>=10 { n/=10;digits+=1; } digits }

fn numerical(g: Geometry<'_>, sequence:u64) -> Result<ForwardNumericalSummary,ModelError> {
    let h=g.hidden;
    let tables=m(h,a(a(g.vocabulary,g.positions)?,g.segments)?)?;
    let block=a(a(m(4,m(h,h)?)?,m(2,m(h,g.intermediate)?)?)?,a(m(9,h)?,g.intermediate)?)?;
    let parameters=a(a(a(tables,m(2,h)?)?,m(g.layers,block)?)?,a(m(h,h)?,h)?)?;
    let parameter_elements=if let Some(c)=g.labels { a(parameters,a(m(c,h)?,c)?)? } else { parameters };
    let elements=|shape:OutputShape| shape.dimensions(g,sequence).iter().try_fold(1,|n,d| m(n,*d));
    let mut block_total=0; let mut largest=0;
    for shape in BLOCK_OUTPUTS { let n=elements(shape)?;block_total=a(block_total,n)?;largest=largest.max(n); }
    let mut total=m(g.layers,block_total)?;
    for shape in EMBEDDING_OUTPUTS.into_iter().chain(POOLER_OUTPUTS) {
        let n=elements(shape)?;total=a(total,n)?;largest=largest.max(n);
    }
    if g.labels.is_some() { let n=elements(OutputShape::Logits)?;total=a(total,n)?;largest=largest.max(n); }
    let public_output_elements=a(a(m(sequence,h)?,h)?,g.labels.unwrap_or(0))?;
    let mut input_bytes=0;let mut largest_input_bytes=0;
    for (_,dtype) in crate::encoder_blocks::FORWARD_INPUTS{
        let width=match dtype {crate::ForwardDType::Int64=>8,crate::ForwardDType::Bool=>1,crate::ForwardDType::Float32=>4};
        let bytes=m(sequence,width)?;input_bytes=a(input_bytes,bytes)?;largest_input_bytes=largest_input_bytes.max(bytes);
    }
    Ok(ForwardNumericalSummary {parameter_elements,operation_output_elements:total,
        largest_operation_elements:largest,input_bytes,largest_input_bytes,public_output_elements})
}

/// Generic source walks and borrowed label comparisons, recomposed rather than
/// pretending decoder bootstrap's fixed lookup/phase fees cover these loops.
fn source_work_for_counts(nodes:usize,payload:usize,bytes:usize)->Result<u64,ModelError>{
    let traversal=source_work::traversal_work_envelope(nodes,payload)
        .map_err(|_|overflow())?;
    // Two native traversals + bounded census replay/headers, lexical scan,
    // source cloning/destruction and borrowed pairwise label/key comparisons.
    let pairs=m(u(nodes)?,u(nodes)?)?;
    let traffic=a(a(u(payload)?,u(bytes)?)?,1)?;
    a(a(m(128,u(traversal)?)?,m(8,u(bytes)?)?)?,m(4,m(pairs,traffic)?)?)
}

pub fn default_forward_limits()->Result<ForwardLimits,ModelError>{
    let policy=pllm_types::resource_limits::MAX_ALLOCATION_BYTES;
    let source_bytes=1_048_576;
    let nodes=65_536;
    let traversal=source_work::traversal_work_envelope(nodes,source_bytes).map_err(|_|overflow())?;
    let mut limits=ForwardLimits {
        source_bytes,
        source:SourceInputLimits {max_nodes:nodes,max_payload_bytes:source_bytes,max_string_bytes:source_bytes,
            max_key_bytes:source_bytes,max_depth:SOURCE_FRAME_SLOTS,max_traversal_work:traversal,
            max_total_work:0,max_workspace_bytes:policy},
        metadata:ConfigLowererLimits {max_operations_per_phase:policy/size_of::<ForwardOperation>(),
            max_string_bytes:source_bytes,max_attribute_nodes:policy/size_of::<Value>(),
            max_retained_strings:policy,max_factory_bytes:policy},
    };
    // Derive, checked, from the same generic traversal/metadata phase equations
    // used below at default maxima. No production Default/u64::MAX/tariff.
    let requirement_limits=requirement_limits(limits.metadata.max_operations_per_phase,
        limits.metadata.max_attribute_nodes,policy,limits.metadata.max_string_bytes)?;
    let q=crate::checkpoint_requirements::quote(requirement_limits).map_err(|_|overflow())?;
    let work=add(q.work_ceiling,forward_requirement_work(limits.metadata.max_operations_per_phase,
        mul(2,limits.metadata.max_operations_per_phase)?,limits.metadata.max_string_bytes)?)?;
    let escaped=add(mul(6,limits.source.max_payload_bytes)?,mul(40,limits.source.max_nodes)?)?;
    let source_work=source_work_for_counts(limits.source.max_nodes,limits.source.max_payload_bytes,limits.source_bytes.max(escaped))?;
    let metadata=metadata_work(source_work,add(limits.metadata.max_attribute_nodes,
        mul(40,limits.metadata.max_operations_per_phase)?)?,policy,policy,work)?;
    limits.source.max_total_work=metadata;
    Ok(limits)
}

/// Caller memory tightens byte/count storage only. Independent finite work
/// ceilings are copied unchanged from the checked default policy, NEVER derived
/// by charging a new bytes-to-work fee. No graph/source ownership is conveyed.
pub fn forward_limits_for_memory(memory_bytes:u64)->Result<ForwardLimits,ModelError>{
    let mut limits=default_forward_limits()?;
    let bytes=usize::try_from(memory_bytes.min(pllm_types::resource_limits::MAX_ALLOCATION_BYTES as u64)).map_err(|_|overflow())?;
    limits.source_bytes=limits.source_bytes.min(bytes);
    limits.source.max_nodes=limits.source.max_nodes.min(bytes/size_of::<Value>());
    limits.source.max_payload_bytes=limits.source.max_payload_bytes.min(bytes);
    limits.source.max_string_bytes=limits.source.max_string_bytes.min(bytes);
    limits.source.max_key_bytes=limits.source.max_key_bytes.min(bytes);
    limits.source.max_workspace_bytes=limits.source.max_workspace_bytes.min(bytes);
    limits.metadata.max_factory_bytes=bytes;
    limits.metadata.max_operations_per_phase=limits.metadata.max_operations_per_phase.min(bytes/size_of::<ForwardOperation>());
    limits.metadata.max_attribute_nodes=limits.metadata.max_attribute_nodes.min(bytes/size_of::<Value>());
    limits.metadata.max_string_bytes=limits.metadata.max_string_bytes.min(bytes);
    limits.metadata.max_retained_strings=limits.metadata.max_retained_strings.min(bytes);
    Ok(limits)
}

fn requirement_limits(operations:usize,nodes:usize,bytes:usize,strings:usize)
    ->Result<crate::checkpoint_requirements::Limits,ModelError>{
    Ok(crate::checkpoint_requirements::Limits {max_operations:operations,max_inputs:mul(2,operations)?,
        max_dimensions:mul(4,operations)?,max_rank:4,max_string_bytes:strings,max_attribute_nodes:nodes,
        max_attribute_bytes:bytes,max_attribute_depth:SOURCE_FRAME_SLOTS+1,max_work:0,max_scratch_bytes:0})
}

fn metadata_workspace(req:&crate::checkpoint_requirements::Quote)->Result<usize,ModelError>{
    let canon=crate::bounded_canonical::resources::<Vec<u8>>(SOURCE_FRAME_SLOTS+6).map_err(|_|overflow())?;
    let native=Traversal::resources_for_native_slots(SOURCE_FRAME_SLOTS).map_err(|_|overflow())?;
    let owned=crate::owned_backing::bounded::resources_for_slots(SOURCE_FRAME_SLOTS).map_err(|_|overflow())?;
    sum(&[req.scratch_bytes,req.attribute_workspace_bytes,canon.hash_workspace_bytes.max(canon.write_workspace_bytes),
        native.logical_workspace_bytes,owned.logical_workspace_bytes,size_of::<ForwardPreflight<'_>>(),
        size_of::<SourceCensus>(),mul(2,size_of::<MetadataLedger>())?,mul(16,size_of::<String>())?,
        mul(3+4+4+3+2+2+4+4,size_of::<ForwardDimension>())?])
}

fn metadata_work(source_work:u64,nodes:usize,strings:usize,canonical:usize,requirements:usize)->Result<u64,ModelError>{
    // Construction/reconstruction/drop + two canonical traversals and hashing.
    // Shared canonical sorting is bounded by byte traffic times total entries;
    // requirements reuse their owner's exact checked comparison/work quote.
    a(source_work,a(u(requirements)?,a(m(32,a(u(nodes)?,u(strings)?)?)?,
        m(4,m(a(u(nodes)?,1)?,a(u(canonical)?,1)?)?)?)?)?)
}

fn forward_requirement_work(ops:usize,records:usize,max_string:usize)->Result<usize,ModelError>{
    let entries=add(ops,records)?;
    let log=(usize::BITS-entries.max(1).leading_zeros()) as usize;
    // Separate new BTreeMap index/lookup/insertion/owned-record loops; the
    // decoder's old quote is NOT claimed to prepay these new consumers.
    mul(mul(mul(32,entries)?,add(log,1)?)?,add(max_string,1)?)
}

fn template_text_bytes()->Result<usize,ModelError>{
    // Union of ALL attribute keys and fixed string literals in encoder_blocks.
    // Summing the entire union per op is conservative and derived from strings,
    // rather than an unexplained per-operation byte allowance.
    let literals=&["weight","bias","weight_shape","bias_shape","weight_layout","formula","compute_dtype",
        "index_domain","out_of_bounds","semantic_config","position_policy","input_layout","layout",
        "group_size","head_dim","kind","mask_input","broadcast","masked_value","application","axis",
        "start","end","squeeze","variance","epsilon","epsilon_placement","affine_order",
        "output_input","x_matmul_weight_transpose_plus_bias","float32","tokens","segments","absolute_positions",
        "reject","explicit_arbitrary_valid","batch_sequence_hidden","batch_heads_sequence_feature",
        "scores_div_sqrt_head_dim","key_padding","input.attention_mask","batch_heads_queries_keys",
        "float32_min","additive","max_subtract_exp_sum","x/2*(1+erf(x/sqrt(2)))",
        "centered_mean_square","inside_sqrt","normalize_multiply_add"];
    literals.iter().try_fold(0,|n,s|add(n,s.len()))
}

struct SourceCensus {backing:usize,nodes:usize,payload:usize,strings:usize,canonical:usize,max_string:usize}

/// DISJOINT reservation slots; every retained subtotal appears exactly once.
/// The sum deliberately dominates the maximum of these sequential phases:
///
/// construct/source digest: source + partial plan + source_digest_tree +
///                         canonical_output + canonical_string + canonical_sort + workspace;
/// validate: source + retained_plan + reconstruction_plan + source_digest_tree +
///           canonical_output + canonical_string + canonical_sort + workspace;
/// plan canonical: source + retained_plan + plan_canonical_tree +
///                 canonical_output + canonical_string + canonical_sort + workspace;
/// requirements: source + retained_plan + requirements + operation_index +
///               artifact_index + workspace (validation's reconstruction is sequential).
///
/// Retained-plan slots INCLUDE their own original source-witness bodies. The
/// source_digest_tree is a distinct serde_json::to_value copy, not another
/// retained witness. Canonical trees are serialized DTO Values, not graph clones.
/// BTree indexes charge concrete shared full-node policy; requirement name/shape
/// bodies move from the artifact index to the output and are charged only in
/// requirements, while both index-node and Vec capacity reservations are summed.
#[derive(Clone,Copy)]
struct MetadataLedger {
    source:usize,retained_plan:usize,reconstruction_plan:usize,source_digest_tree:usize,
    plan_canonical_tree:usize,canonical_output:usize,canonical_string:usize,canonical_sort:usize,
    requirements:usize,workspace:usize,operation_index:usize,artifact_index:usize,
}
impl MetadataLedger {
    fn peak(self)->Result<usize,ModelError>{sum(&[self.source,self.retained_plan,self.reconstruction_plan,
        self.source_digest_tree,self.plan_canonical_tree,self.canonical_output,self.canonical_string,
        self.canonical_sort,self.requirements,self.workspace,self.operation_index,self.artifact_index])}
}
fn escaped(text:&str)->Result<usize,ModelError>{
    text.bytes().try_fold(2,|n,b|add(n,match b {b'"'|b'\\'|b'\n'|b'\r'|b'\t'|8|12=>2,0..=31=>6,_=>1}))
}
fn census(source:&Value,l:ForwardLimits)->Result<SourceCensus,ModelError>{
    let mut native=source_work::traversal_limits(l.source).map_err(|_|invalid("forward source traversal limits"))?;
    native.max_work=native.max_work.min(usize::try_from(l.source.max_total_work).unwrap_or(usize::MAX));
    let owned=crate::owned_backing::bounded::resources_for_slots(SOURCE_FRAME_SLOTS).map_err(|_|overflow())?;
    let canonical=crate::bounded_canonical::resources::<Vec<u8>>(SOURCE_FRAME_SLOTS+6).map_err(|_|overflow())?;
    let workspace=sum(&[owned.logical_workspace_bytes,native.max_workspace_bytes,
        canonical.write_workspace_bytes,size_of::<ForwardPreflight<'_>>(),size_of::<SourceCensus>()])?;
    if workspace>l.source.max_workspace_bytes || workspace>l.metadata.max_factory_bytes {
        return Err(invalid("forward source workspace limit"));
    }
    let mut frames=[const {NativeFrame::empty()};SOURCE_FRAME_SLOTS];
    let mut walk=Traversal::new_native(source,&mut frames,native).map_err(|_|invalid("forward source census"))?;
    let mut c=SourceCensus {backing:0,nodes:0,payload:0,strings:0,canonical:0,max_string:0};
    while let Some(event)=walk.next_event().map_err(|_|invalid("forward source census limit"))? {
        match event {
            Event::Key{key,..}=>{ c.strings=add(c.strings,key.len())?;c.max_string=c.max_string.max(key.len());c.canonical=add(c.canonical,add(escaped(key)?,1)?)?; }
            Event::Value{value,number_bytes,..}=>{
                c.canonical=add(c.canonical,match value {
                    Value::String(s)=>{c.strings=add(c.strings,s.len())?;c.max_string=c.max_string.max(s.len());escaped(s)?},
                    Value::Array(a)=>add(2,a.len().saturating_sub(1))?,
                    Value::Object(o)=>add(2,o.len().saturating_sub(1))?,
                    Value::Number(_)=>number_bytes,Value::Bool(true)=>4,Value::Bool(false)=>5,Value::Null=>4,
                })?;
            }
        }
    }
    let usage=walk.usage(); c.nodes=usage.nodes;c.payload=usage.bytes;
    drop(walk);
    let work=source_work::traversal_work_envelope(c.nodes,c.payload).map_err(|_|overflow())?;
    let owner_work=mul(64,work)?;
    if a(u(usage.work)?,u(owner_work)?)?>l.source.max_total_work{return Err(invalid("forward source backing work limit"));}
    let mut owner=crate::owned_backing::bounded::Census::new(&mut frames,crate::owned_backing::bounded::Limits {
        max_backing_bytes:l.metadata.max_factory_bytes,max_work:owner_work,
        max_workspace_bytes:owned.logical_workspace_bytes,max_json_nodes:mul(2,c.nodes)?,
        max_json_payload_bytes:mul(2,c.payload)?,max_json_string_bytes:l.source.max_string_bytes,
        max_json_key_bytes:l.source.max_key_bytes,max_json_depth:l.source.max_depth-1,
    }).map_err(|_|invalid("forward source backing workspace"))?;
    c.backing=owner.json(source).map_err(|_|invalid("forward source backing limit"))?.backing_bytes();
    Ok(c)
}

pub fn preflight_forward_value<'a>(source:&'a Value,max_sequence:u64,limits:ForwardLimits)
    ->Result<ForwardPreflight<'a>,ModelError>{
    let c=census(source,limits)?;
    if source_work_for_counts(c.nodes,c.payload,c.canonical)?>limits.source.max_total_work {
        return Err(invalid("forward borrowed source comparison work limit"));
    }
    // Generic census and work admission precede even borrowed scalar/label
    // validation. No layer expansion, name formatting, clone or digest here.
    let (registry,g)=crate::source_mappings::forward_geometry(source,max_sequence)?;
    let quote=quote_geometry(g,max_sequence,registry,&c,limits)?;
    Ok(ForwardPreflight {source,sequence:max_sequence,geometry:g,registry,limits,quote})
}

fn quote_geometry(g:Geometry<'_>,sequence:u64,registry:&EncoderSource,c:&SourceCensus,l:ForwardLimits)
    ->Result<ForwardResourceQuote,ModelError>{
    quote_geometry_with_ledger(g,sequence,registry,c,l).map(|(quote,_)|quote)
}

fn quote_geometry_with_ledger(g:Geometry<'_>,sequence:u64,registry:&EncoderSource,c:&SourceCensus,l:ForwardLimits)
    ->Result<(ForwardResourceQuote,MetadataLedger),ModelError>{
    let classifier=usize::from(g.labels.is_some());
    let layers=usize::try_from(g.layers).map_err(|_|overflow())?;
    let ops=add(mul(BLOCK_OUTPUTS.len(),layers)?,add(EMBEDDING_OUTPUTS.len()+POOLER_OUTPUTS.len(),classifier)?)?;
    let records=add(mul(16,layers)?,add(7,mul(2,classifier)?)?)?;
    if ops>l.metadata.max_operations_per_phase {return Err(invalid("forward operation limit before expansion"));}
    let name=registry.maximum_artifact_bytes(g.labels.is_some(),digits(g.layers-1))?.max("layer..attention_residual".len()+digits(g.layers-1));
    let max_string=name.max(c.max_string).max("x_matmul_weight_transpose_plus_bias".len()).max(48);
    if max_string>l.metadata.max_string_bytes {return Err(invalid("forward string limit"));}
    // The closed builder has <=10 attribute members, <=2 shape arrays of rank
    // <=2, <=2 inputs/rank<=4, <=7 string values. These are structural upper
    // bounds on current templates, not estimated bytes per arbitrary JSON node.
    let attribute_nodes=add(mul(16,ops)?,c.nodes)?;
    let text=template_text_bytes()?;
    // 12 name-bearing slots cover id, two input names, artifact identities and
    // transient formatting growth; epsilon has <=48 bytes after FP32 normalize.
    let strings=add(mul(ops,add(add(mul(12,name)?,text)?,48)?)?,add(c.strings,256)?)?;
    if attribute_nodes>l.metadata.max_attribute_nodes || strings>l.metadata.max_retained_strings {
        return Err(invalid("forward attributes/retained strings limit"));
    }
    let attribute_bytes=add(strings,mul(32,attribute_nodes)?)?;
    let node=tree::<(String,Value)>()?;
    // Input iterator collection may retain Vec's four-element initial capacity
    // even for unary/binary operations. Output shapes use exact vec! ranks.
    let op_backing=sum(&[size_of::<ForwardOperation>(),mul(10,node)?,mul(4,size_of::<String>())?,
        mul(4,size_of::<ForwardDimension>())?,mul(4,size_of::<Value>())?])?;
    let maps=add(mul(4,tree::<(String,ForwardTensorSpec)>()?)?,mul(3,tree::<(String,ForwardOutput)>()?)?)?;
    let root_strings=["pllm.forward_plan.v1","pllm.numeric.source_float32.v1","pllm.forward.encoder_outputs.v1",
        registry.family,registry.adapter,"input.tokens","input.positions","input.segments","input.attention_mask",
        "last_hidden_state","pooler_output","logits"];
    let root_text=root_strings.iter().try_fold(64,|n,s|add(n,s.len()))?;
    let root_shapes=mul(7*4,size_of::<ForwardDimension>())?;
    let plan=sum(&[size_of::<ForwardPlan>(),mul(ops,op_backing)?,strings,maps,c.backing,root_text,root_shapes])?;
    let requirement_record=add(size_of::<ForwardRequirement>(),add(name,mul(3,size_of::<u64>())?)?)?;
    let retained_requirements_bytes=add(size_of::<Vec<ForwardRequirement>>(),mul(records,requirement_record)?)?;
    let req_limits=requirement_limits(ops,attribute_nodes,attribute_bytes,max_string)?;
    let req=crate::checkpoint_requirements::quote(req_limits).map_err(|_|overflow())?;
    let workspace=metadata_workspace(&req)?;
    if workspace>l.source.max_workspace_bytes {return Err(invalid("forward metadata workspace limit"));}
    // Exact escaped source bytes from census; closed typed schema upper bound:
    // <=40 serialized nodes/op, bounded fixed keys, names and integer spellings.
    // <=40 typed+attribute values/op; every number has <=20 decimal digits,
    // every fixed key/value belongs to the above union, and dynamic names are
    // ASCII. Punctuation/quotes consume <=4 bytes per schema value/member.
    let per_op_canonical=sum(&[mul(2,text)?,mul(24,name)?,mul(40,24)?,mul(2,48)?])?;
    let canonical_bytes=add(c.canonical,add(mul(ops,per_op_canonical)?,add(mul(64,24)?,mul(2,root_text)?)?)?)?;
    let serialized_tree=sum(&[size_of::<Value>(),mul(add(mul(40,ops)?,64)?,node)?,strings,c.backing])?;
    // Ordinary Vec geometric growth <=2*final bytes for these nonempty >128B
    // records. String serializer staging starts at 128B and is <=2*whole-record
    // canonical bytes. Both source and plan canonical phases reuse these slots.
    // Sorting retains ancestor entry Vecs while descending. <=4 initial slots
    // per frame cover small Vec capacities; the second copy covers sort scratch.
    let sort_entries=add(add(c.nodes,12+4+3+2+2+6+10)?,mul(4,SOURCE_FRAME_SLOTS+6)?)?;
    let ledger=MetadataLedger {source:c.backing,retained_plan:plan,reconstruction_plan:plan,
        source_digest_tree:add(size_of::<Value>(),c.backing)?,plan_canonical_tree:serialized_tree,
        canonical_output:mul(2,canonical_bytes)?,canonical_string:mul(2,canonical_bytes)?,
        canonical_sort:mul(mul(2,sort_entries)?,size_of::<(&String,&Value)>())?,
        requirements:retained_requirements_bytes,workspace,
        operation_index:mul(ops,tree::<(&str,u64)>()?)?,artifact_index:mul(records,tree::<(String,Vec<u64>)>()?)?};
    let peak=ledger.peak()?;
    let req_work=add(req.work_ceiling,forward_requirement_work(ops,records,max_string)?)?;
    let source_work=source_work_for_counts(c.nodes,c.payload,c.canonical)?;
    let work=metadata_work(source_work,add(attribute_nodes,mul(40,ops)?)?,strings,canonical_bytes,req_work)?;
    if peak>l.metadata.max_factory_bytes {return Err(invalid("forward metadata peak limit before expansion"));}
    if work>l.source.max_total_work {return Err(invalid("forward metadata work limit before expansion"));}
    Ok((ForwardResourceQuote {operation_count:ops,operation_id_bytes:crate::encoder_blocks::operation_id_bytes(g)?,
        source_backing_bytes:c.backing,retained_plan_bytes:plan,
        retained_requirements_bytes,metadata_peak_bytes:peak,metadata_work:work,numerical:numerical(g,sequence)?},ledger))
}

pub fn lower_forward_model_value_with_limits(source:&Value,sequence:u64,limits:ForwardLimits)->Result<ForwardPlan,ModelError>{
    preflight_forward_value(source,sequence,limits)?.construct()
}

pub fn lower_forward_model_with_limits(source:&[u8],sequence:u64,limits:ForwardLimits)->Result<ForwardPlan,ModelError>{
    let document=parse_forward_source_json(source,limits)?;
    lower_forward_model_value_with_limits(&document,sequence,limits)
}

/// Model-owned reusable source parser. Lexical and duplicate/backing admission
/// precede Value allocation. Exact-decimal field selection stays in the source
/// registry, not in native/Python consumers.
pub fn parse_forward_source_json(source:&[u8],limits:ForwardLimits)->Result<Value,ModelError>{
    crate::forward_plan::parse_bounded_source_json(source,limits,crate::source_mappings::FORWARD_EXACT_DECIMALS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{forward_checkpoint_requirements,lower_forward_model,lower_forward_model_value};
    use serde_json::json;

    fn fixture()->Value{json!({"model_type":"bert","architectures":["BertForSequenceClassification"],
        "hidden_size":8,"num_hidden_layers":2,"num_attention_heads":2,"intermediate_size":13,
        "vocab_size":17,"max_position_embeddings":8,"type_vocab_size":2,"hidden_act":"gelu",
        "position_embedding_type":"absolute","layer_norm_eps":1e-12,"num_labels":3})}

    #[test]
    fn borrowed_quote_precedes_construct_and_reconciles_every_tensor_and_output(){
        let source=fixture();let limits=default_forward_limits().unwrap();
        let owner=preflight_forward_value(&source,5,limits).unwrap();let q=*owner.quote();
        let plan=owner.construct().unwrap();let requirements=forward_checkpoint_requirements(&plan).unwrap();
        assert_eq!(requirements.len(),41);
        let parameters= requirements.iter().map(|r|r.shape.iter().product::<u64>()).sum::<u64>();
        assert_eq!(q.numerical.parameter_elements,parameters);
        // Generated P=8/I=13 fixture: 1429 elements. The distinct contract
        // P=16/I=12 fixture has 1459 elements; both have 41 tensors.
        assert_eq!(parameters*4,5716);
        let outputs=plan.operations.iter().map(|op|op.output.shape.iter().map(|d|match d{
            ForwardDimension::Fixed(n)=>*n,ForwardDimension::Sequence=>5}).product::<u64>()).collect::<Vec<_>>();
        assert_eq!(outputs.iter().sum::<u64>(),q.numerical.operation_output_elements);
        assert_eq!(q.numerical.operation_output_elements,2047);
        assert_eq!(q.numerical.largest_operation_elements,65);
        assert_eq!(*outputs.iter().max().unwrap(),q.numerical.largest_operation_elements);
        assert_eq!(q.numerical.input_bytes,125);
        assert_eq!(q.numerical.largest_input_bytes,40);
        assert_eq!(q.operation_count,plan.operations.len());
        assert_eq!(q.operation_id_bytes,plan.operations.iter().map(|op|op.id.len()).sum::<usize>());
        assert_eq!(q.numerical.public_output_elements,51);
        assert!(plan.checked_owned_backing_bytes(limits.metadata.max_factory_bytes).unwrap()<=q.retained_plan_bytes);
        let actual_requirements=std::mem::size_of_val(&requirements)+requirements.capacity()*size_of::<ForwardRequirement>()+
            requirements.iter().map(|r|r.name.capacity()+r.shape.capacity()*size_of::<u64>()).sum::<usize>();
        assert!(actual_requirements<=q.retained_requirements_bytes);
        assert!(pllm_types::canonical_bytes(&plan).len()<=q.metadata_peak_bytes);
    }

    #[test]
    fn exact_and_one_short_peak_work_operations_attributes_strings(){
        let source=fixture();let base=default_forward_limits().unwrap();
        let quote=*preflight_forward_value(&source,5,base).unwrap().quote();
        let mut exact=base;exact.metadata.max_factory_bytes=quote.metadata_peak_bytes;
        exact.source.max_total_work=quote.metadata_work;
        assert!(preflight_forward_value(&source,5,exact).is_ok());
        let mut short=exact;short.metadata.max_factory_bytes-=1;
        assert!(preflight_forward_value(&source,5,short).is_err());
        let mut short=exact;short.source.max_total_work-=1;
        assert!(preflight_forward_value(&source,5,short).is_err());
        let mut ops=base;ops.metadata.max_operations_per_phase=50;
        assert!(preflight_forward_value(&source,5,ops).is_ok());
        ops.metadata.max_operations_per_phase=49;
        assert!(preflight_forward_value(&source,5,ops).is_err());
        let mut workspace=base;workspace.source.max_workspace_bytes=0;
        assert!(preflight_forward_value(&source,5,workspace).is_err());
        let mut attributes=base;attributes.metadata.max_attribute_nodes=0;
        assert!(preflight_forward_value(&source,5,attributes).is_err());
        let mut strings=base;strings.metadata.max_retained_strings=0;
        assert!(preflight_forward_value(&source,5,strings).is_err());
    }

    #[test]
    fn byte_parser_limits_are_upfront_duplicate_aware_and_exact_decimal(){
        let source=fixture();let bytes=serde_json::to_vec(&source).unwrap();let mut limits=default_forward_limits().unwrap();
        limits.source_bytes=bytes.len();
        let parsed=parse_forward_source_json(&bytes,limits).unwrap();
        assert_eq!(lower_forward_model_with_limits(&bytes,5,limits).unwrap(),lower_forward_model_value(&parsed,5).unwrap());
        limits.source_bytes-=1;
        assert!(lower_forward_model_with_limits(&bytes,5,limits).is_err());
        for duplicate in [br#"{"a":1,"a":2}"#.as_slice(),br#"{"a":1,"\u0061":2}"#.as_slice(),br#"{"x":{"a":1,"a":1}}"#.as_slice()]{
            assert!(parse_forward_source_json(duplicate,default_forward_limits().unwrap()).is_err());
        }
        let base=default_forward_limits().unwrap();
        let nested=b"[[0]]";let mut exact=base;exact.source.max_depth=3;
        assert!(parse_forward_source_json(nested,exact).is_ok());exact.source.max_depth=2;
        assert!(parse_forward_source_json(nested,exact).is_err());
        let mut nodes=base;nodes.source.max_nodes=3;
        assert!(parse_forward_source_json(nested,nodes).is_ok());nodes.source.max_nodes=2;
        assert!(parse_forward_source_json(nested,nodes).is_err());
        let mut tiny=base;tiny.metadata.max_factory_bytes=1;
        assert!(parse_forward_source_json(&bytes,tiny).is_err());
        let raw="1.0000000596046447753906250000001";
        let mut document=fixture();document["layer_norm_eps"]=json!(raw);
        let text=serde_json::to_string(&document).unwrap().replace("\"1.0000000596046447753906250000001\"",raw);
        let p=lower_forward_model(text.as_bytes(),5).unwrap();
        let epsilon=p.operations.iter().find(|op|op.id=="embedding.norm").unwrap().attributes["epsilon"].as_str().unwrap();
        assert_eq!(crate::parse_positive_f32(epsilon).unwrap().to_bits(),1f32.to_bits()+1);
        assert_ne!(crate::parse_positive_f32(epsilon).unwrap().to_bits(),(raw.parse::<f64>().unwrap()as f32).to_bits());
        let direct=lower_forward_model_value(&document,5).unwrap();
        assert_eq!(direct,p);
    }

    #[test]
    fn escaped_source_census_is_exact_and_old_value_capacity_is_separate(){
        let mut source=fixture();source["_name_or_path"]=json!("x\u{0}\n\\\"é");
        let limits=default_forward_limits().unwrap();let c=census(&source,limits).unwrap();
        assert_eq!(c.canonical,pllm_types::canonical_bytes(&source).len());
        let q=*preflight_forward_value(&source,5,limits).unwrap().quote();
        assert_eq!(q.source_backing_bytes,c.backing);
        assert!(q.metadata_peak_bytes>q.source_backing_bytes+q.retained_plan_bytes);
    }

    #[test]
    fn quote_is_cap_independent_and_memory_derivation_keeps_work_policy(){
        let source=fixture();let base=default_forward_limits().unwrap();
        let q=*preflight_forward_value(&source,5,base).unwrap().quote();
        let exact=forward_limits_for_memory(q.metadata_peak_bytes as u64).unwrap();
        assert_eq!(exact.source.max_total_work,base.source.max_total_work);
        assert_eq!(exact.source.max_traversal_work,base.source.max_traversal_work);
        assert_eq!(q,*preflight_forward_value(&source,5,exact).unwrap().quote());
        let short=forward_limits_for_memory(q.metadata_peak_bytes as u64-1).unwrap();
        assert!(preflight_forward_value(&source,5,short).is_err());
        let mut tighter=base;tighter.source.max_nodes=1024;tighter.source.max_payload_bytes=4096;
        tighter.metadata.max_operations_per_phase=50;tighter.metadata.max_string_bytes=128;
        assert_eq!(q,*preflight_forward_value(&source,5,tighter).unwrap().quote());
    }

    #[test]
    fn exact_and_one_short_workspace_payload_nodes_and_traversal_work(){
        let source=fixture();let base=default_forward_limits().unwrap();let c=census(&source,base).unwrap();
        let (registry,g)=crate::source_mappings::forward_geometry(&source,5).unwrap();
        let name=registry.maximum_artifact_bytes(true,digits(g.layers-1)).unwrap().max("layer..attention_residual".len()+digits(g.layers-1));
        let max_string=name.max(c.max_string).max("x_matmul_weight_transpose_plus_bias".len()).max(48);
        let nodes=50*16+c.nodes;
        let strings=50*(12*name+template_text_bytes().unwrap()+48)+c.strings+256;
        let req=crate::checkpoint_requirements::quote(requirement_limits(50,nodes,strings+32*nodes,max_string).unwrap()).unwrap();
        let workspace=metadata_workspace(&req).unwrap();
        let mut exact=base;exact.source.max_workspace_bytes=workspace;
        assert!(preflight_forward_value(&source,5,exact).is_ok());
        exact.source.max_workspace_bytes-=1;
        assert!(preflight_forward_value(&source,5,exact).is_err());
        let mut exact=base;exact.source.max_payload_bytes=c.payload;exact.source.max_nodes=c.nodes;
        assert!(preflight_forward_value(&source,5,exact).is_ok());
        let mut short=exact;short.source.max_nodes-=1;
        assert!(preflight_forward_value(&source,5,short).is_err());
        let mut short=exact;short.source.max_payload_bytes-=1;
        assert!(preflight_forward_value(&source,5,short).is_err());
        let mut frames=[const {NativeFrame::empty()};SOURCE_FRAME_SLOTS];
        let mut walker=Traversal::new_native(&source,&mut frames,source_work::traversal_limits(base.source).unwrap()).unwrap();
        while walker.next_event().unwrap().is_some(){}
        let work=walker.usage().work;
        let mut exact=base;exact.source.max_traversal_work=work;
        assert!(preflight_forward_value(&source,5,exact).is_ok());exact.source.max_traversal_work-=1;
        assert!(preflight_forward_value(&source,5,exact).is_err());
    }

    #[test]
    fn descriptor_id_accounting_spans_decimal_index_boundaries(){
        let mut source=fixture();source["num_hidden_layers"]=json!(11);
        let owner=preflight_forward_value(&source,1,default_forward_limits().unwrap()).unwrap();
        let q=*owner.quote();let plan=owner.construct().unwrap();
        assert_eq!(q.operation_id_bytes,plan.operations.iter().map(|op|op.id.len()).sum::<usize>());
    }

    #[test]
    fn original_epsilon_witness_and_digest_survive_semantic_normalization(){
        let limits=default_forward_limits().unwrap();let mut previous=None;
        for spelling in ["1e-12","0.000000000001","1.0000000596046447753906250000001"]{
            let mut source=fixture();source["layer_norm_eps"]=json!(spelling);
            let (registry,_)=crate::source_mappings::forward_geometry(&source,5).unwrap();
            let plan=preflight_forward_value(&source,5,limits).unwrap().construct().unwrap();
            assert_eq!(plan.operations[0].attributes["semantic_config"],source);
            assert_eq!(plan.config_digest,pllm_types::canonical_digest(registry.digest_domain,&source));
            let normalized=crate::parse_positive_f32(spelling).unwrap().to_string();
            if spelling!="1.0000000596046447753906250000001"{
                assert_eq!(normalized,"0.000000000001");
            }
            for op in plan.operations.iter().filter(|op|op.operator==crate::ModelOperator::LayerNorm){
                assert_eq!(op.attributes["epsilon"].as_str(),Some(normalized.as_str()));
            }
            plan.validate_with_limits(limits).unwrap();
            let bytes=serde_json::to_vec(&source).unwrap();
            assert_eq!(lower_forward_model_with_limits(&bytes,5,limits).unwrap(),plan);
            if spelling=="1e-12"{previous=Some(plan.config_digest.clone());}
            if spelling=="0.000000000001"{assert_ne!(previous.as_ref().unwrap(),&plan.config_digest);}
        }
    }

    #[test]
    fn custom_policy_survives_construct_validation_and_requirements_above_default_peak(){
        let base=default_forward_limits().unwrap();let mut custom=base;
        custom.metadata.max_factory_bytes=base.metadata.max_factory_bytes.checked_mul(4).unwrap();
        // Small H/I, scalar-only source. Select a metadata expansion crossing
        // 64MiB; no weight/activation arrays or guessed architecture ceiling.
        let mut source=fixture();let mut selected=None;
        for layers in [16,32,64,128,256]{
            source["num_hidden_layers"]=json!(layers);
            let q=*preflight_forward_value(&source,1,custom).unwrap().quote();
            if q.metadata_peak_bytes>base.metadata.max_factory_bytes{selected=Some(q);break;}
        }
        let q=selected.expect("fixture crosses default metadata allocation policy");
        assert!(q.metadata_peak_bytes>pllm_types::resource_limits::MAX_ALLOCATION_BYTES);
        assert!(preflight_forward_value(&source,1,base).is_err());
        let plan=preflight_forward_value(&source,1,custom).unwrap().construct().unwrap();
        assert!(plan.validate().is_err());assert!(forward_checkpoint_requirements(&plan).is_err());
        plan.validate_with_limits(custom).unwrap();
        let requirements=crate::forward_checkpoint_requirements_with_limits(&plan,custom).unwrap();
        assert_eq!(requirements.len()as u64,16*source["num_hidden_layers"].as_u64().unwrap()+9);
        let wire=serde_json::to_vec(&plan).unwrap();
        let replay:ForwardPlan=serde_json::from_slice(&wire).unwrap();
        replay.validate_with_limits(custom).unwrap();
        assert_eq!(crate::forward_checkpoint_requirements_with_limits(&replay,custom).unwrap(),requirements);
        assert_eq!(replay.digest(),plan.digest()); // identity APIs impose no hidden default reconstruction
    }

    #[test]
    fn contract_fixture_and_generated_fixture_have_distinct_correct_parameter_totals(){
        let mut source=fixture();source["max_position_embeddings"]=json!(16);source["intermediate_size"]=json!(12);
        let owner=preflight_forward_value(&source,5,default_forward_limits().unwrap()).unwrap();
        let q=*owner.quote();let plan=owner.construct().unwrap();
        let requirements=forward_checkpoint_requirements(&plan).unwrap();
        assert_eq!(requirements.len(),41);
        let actual=requirements.iter().map(|r|r.shape.iter().product::<u64>()).sum::<u64>();
        assert_eq!(actual,1459);assert_eq!(q.numerical.parameter_elements*4,5836);
    }

    // Observe ordinary canonical writer allocations without a global allocator:
    // actual entry-Vec capacities along the longest live ancestor chain and
    // actual serde escaped-string/number buffer capacities. Twice entry backing
    // conservatively covers sorting scratch in addition to retained entry Vecs.
    fn observed_canonical_staging(value:&Value)->(usize,usize){
        match value{
            Value::Object(map)=>{
                let entries:Vec<_>=map.iter().collect();
                let own=2*entries.capacity()*size_of::<(&String,&Value)>();
                let mut sort=0;let mut escaped=0;
                for (key,value) in entries{
                    escaped=escaped.max(serde_json::to_vec(key).unwrap().capacity());
                    let (child_sort,child_escaped)=observed_canonical_staging(value);
                    sort=sort.max(child_sort);escaped=escaped.max(child_escaped);
                }
                (own+sort,escaped)
            }
            Value::Array(array)=>array.iter().map(observed_canonical_staging)
                .fold((0,0),|(s,e),(a,b)|(s.max(a),e.max(b))),
            Value::String(s)=>(0,serde_json::to_vec(s).unwrap().capacity()),
            Value::Number(n)=>(0,n.to_string().capacity()),
            _=>(0,0),
        }
    }

    #[test]
    fn disjoint_phase_ledger_bounds_independent_capacities_and_reconstruction_overlap(){
        let mut source=fixture();source["_name_or_path"]=json!("x\u{0}\n\\\"é".repeat(128));
        source["layer_norm_eps"]=json!("1e-12");
        let limits=default_forward_limits().unwrap();let c=census(&source,limits).unwrap();
        let (registry,g)=crate::source_mappings::forward_geometry(&source,5).unwrap();
        let (q,ledger)=quote_geometry_with_ledger(g,5,registry,&c,limits).unwrap();
        assert_eq!(q.metadata_peak_bytes,ledger.peak().unwrap());
        let plan=preflight_forward_value(&source,5,limits).unwrap().construct().unwrap();
        let witness=&plan.operations[0].attributes["semantic_config"];
        let reconstruction=preflight_forward_value(witness,5,limits).unwrap().construct().unwrap();
        assert_eq!(plan,reconstruction);
        let bound=limits.metadata.max_factory_bytes;
        let source_body=crate::owned_backing::checked_json_backing_bytes(&source,bound).unwrap();
        let plan_body=plan.checked_owned_backing_bytes(bound).unwrap();
        let reconstruction_body=reconstruction.checked_owned_backing_bytes(bound).unwrap();
        assert!(source_body<=ledger.source);
        assert!(plan_body<=ledger.retained_plan);assert!(reconstruction_body<=ledger.reconstruction_plan);
        assert_eq!(plan.operations.capacity(),q.operation_count);
        let ids=plan.operations.iter().map(|op|op.id.capacity()).sum::<usize>();
        let input_strings=plan.operations.iter().flat_map(|op|&op.inputs).map(String::capacity).sum::<usize>();
        assert!(ids+input_strings<=ledger.retained_plan);
        let source_tree=serde_json::to_value(&source).unwrap();
        let plan_tree=serde_json::to_value(&plan).unwrap();
        let source_tree_bytes=size_of::<Value>()+crate::owned_backing::checked_json_backing_bytes(&source_tree,bound).unwrap();
        let plan_tree_bytes=size_of::<Value>()+crate::owned_backing::checked_json_backing_bytes(&plan_tree,bound).unwrap();
        assert!(source_tree_bytes<=ledger.source_digest_tree);assert!(plan_tree_bytes<=ledger.plan_canonical_tree);
        let source_canon=pllm_types::canonical_bytes(&source);
        let plan_canon=pllm_types::canonical_bytes(&plan);
        let (source_sort,source_escape)=observed_canonical_staging(&source_tree);
        let (plan_sort,plan_escape)=observed_canonical_staging(&plan_tree);
        assert!(source_canon.capacity().max(plan_canon.capacity())<=ledger.canonical_output);
        assert!(source_sort.max(plan_sort)<=ledger.canonical_sort);
        assert!(source_escape.max(plan_escape)<=ledger.canonical_string);
        let requirements=crate::forward_checkpoint_requirements_with_limits(&plan,limits).unwrap();
        let req_bytes=std::mem::size_of_val(&requirements)+requirements.capacity()*size_of::<ForwardRequirement>()+
            requirements.iter().map(|r|r.name.capacity()+r.shape.capacity()*size_of::<u64>()).sum::<usize>();
        assert!(req_bytes<=ledger.requirements);
        // Index bodies move into the result; only disjoint index node backing
        // is added to the measured retained requirement bodies/Vec here.
        let op_index=plan.operations.len()*tree::<(&str,u64)>().unwrap();
        let artifact_index=requirements.len()*tree::<(String,Vec<u64>)>().unwrap();
        assert!(op_index<=ledger.operation_index);assert!(artifact_index<=ledger.artifact_index);
        let construct=source_body+plan_body+source_tree_bytes+source_canon.capacity()+source_escape+source_sort+ledger.workspace;
        let validate=source_body+plan_body+reconstruction_body+source_tree_bytes+
            source_canon.capacity()+source_escape+source_sort+ledger.workspace;
        let canonical=source_body+plan_body+plan_tree_bytes+plan_canon.capacity()+plan_escape+plan_sort+ledger.workspace;
        let requirements_phase=source_body+plan_body+req_bytes+op_index+artifact_index+ledger.workspace;
        assert!(construct.max(validate).max(canonical).max(requirements_phase)<=q.metadata_peak_bytes);
    }

    #[test]
    fn source_sequence_binding_null_label_policy_and_huge_counts(){
        let mut source=fixture();source["id2label"]=Value::Null;source["label2id"]=Value::Null;
        let limits=default_forward_limits().unwrap();
        let first=preflight_forward_value(&source,4,limits).unwrap();
        let second=preflight_forward_value(&source,5,limits).unwrap();
        assert_ne!(first.quote().numerical,second.quote().numerical);
        assert_eq!(first.construct().unwrap().max_sequence,4);
        source.as_object_mut().unwrap().remove("num_labels");
        assert!(preflight_forward_value(&source,5,limits).is_err());
        source["architectures"]=json!(["BertModel"]);
        assert!(preflight_forward_value(&source,5,limits).is_ok());
        source["num_labels"]=Value::Null;
        assert!(preflight_forward_value(&source,5,limits).is_err());
        source["architectures"]=json!(["BertForSequenceClassification"]);
        source["num_labels"]=json!(3);source["num_hidden_layers"]=json!(u64::MAX/64);
        assert!(preflight_forward_value(&source,5,limits).is_err());
        source["num_hidden_layers"]=json!(2);source["hidden_size"]=json!(u64::MAX-1);
        assert!(preflight_forward_value(&source,5,limits).is_err());
    }
}
