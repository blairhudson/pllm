//! Stateless, batch-one source FP32 forward plans. These descriptive plans grant
//! no execution, placement, disclosure, or protected-source admission authority.
//!
//! Runtime contract: all four [1,S] inputs are mandatory, 1 <= S <= max_sequence.
//! Tokens/positions/segments are signed int64 indices, nonnegative and below
//! their declared embedding cardinalities. Positions may be arbitrary valid
//! indices. Attention masks are bool and must contain at least one visible key.
//! Defaults belong to the SDK, never graph inference. Dropout is eval identity.
//! AttentionScale divides scores by sqrt(D), with conversion, sqrt and division
//! in FP32 (not multiplication by a separately rounded reciprocal). Softmax is
//! max-subtracted FP32 exp/sum/divide. No hidden intermediate capture is allowed.
use crate::{ModelError, ModelOperator};
use pllm_types::{canonical_digest, Digest};
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const FORWARD_PLAN_SCHEMA_VERSION: &str = "pllm.forward_plan.v1";
pub const FORWARD_NUMERIC: &str = "pllm.numeric.source_float32.v1";
pub const FORWARD_OUTPUT_CONTRACT: &str = "pllm.forward.encoder_outputs.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForwardDimension {
    Fixed(u64),
    Sequence,
}

impl Serialize for ForwardDimension {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Fixed(n) if *n > 0 => s.serialize_u64(*n),
            Self::Fixed(_) => Err(serde::ser::Error::custom("zero dimension")),
            Self::Sequence => s.serialize_str("sequence"),
        }
    }
}
impl<'de> Deserialize<'de> for ForwardDimension {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Dimension;
        impl<'de> de::Visitor<'de> for Dimension {
            type Value = ForwardDimension;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("positive integer or sequence")
            }
            fn visit_u64<E: de::Error>(self, n: u64) -> Result<Self::Value, E> {
                if n == 0 { Err(E::custom("zero dimension")) } else { Ok(ForwardDimension::Fixed(n)) }
            }
            fn visit_str<E: de::Error>(self, s: &str) -> Result<Self::Value, E> {
                if s == "sequence" { Ok(ForwardDimension::Sequence) } else { Err(E::custom("unknown dimension")) }
            }
        }
        d.deserialize_any(Dimension)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ForwardDType { Int64, Bool, Float32 }

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardTensorSpec { pub dtype: ForwardDType, pub shape: Vec<ForwardDimension> }

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardOperation {
    pub id: String,
    pub operator: ModelOperator,
    pub layer: Option<u64>,
    pub inputs: Vec<String>,
    pub output: ForwardTensorSpec,
    pub attributes: Value,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardOutput { pub value: String, pub tensor: ForwardTensorSpec }

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardPlan {
    pub schema_version: String,
    pub model_family: String,
    pub adapter: String,
    pub config_digest: Digest,
    pub max_sequence: u64,
    pub numeric: String,
    pub output_contract: String,
    pub inputs: BTreeMap<String, ForwardTensorSpec>,
    pub operations: Vec<ForwardOperation>,
    pub outputs: BTreeMap<String, ForwardOutput>,
}

impl ForwardPlan {
    /// SHA256(domain || NUL || canonical JSON), same shared canonical primitive
    /// as existing plans. Identity is not source provenance or admission.
    pub fn digest(&self) -> Digest { canonical_digest(FORWARD_PLAN_SCHEMA_VERSION, self) }

    /// Existing logical tree/capacity policy, for observation/regression tests;
    /// not construction authority and not an allocator/RSS measurement.
    pub fn checked_owned_backing_bytes(&self,max_bytes:usize)->Result<usize,ModelError>{
        use crate::forward_resources::{add,mul};
        let fail=||invalid("forward owned backing limit");
        let mut bytes=std::mem::size_of::<Self>();
        for s in [&self.schema_version,&self.model_family,&self.adapter,&self.numeric,&self.output_contract]{bytes=add(bytes,s.capacity())?;}
        bytes=add(bytes,64)?;
        bytes=add(bytes,mul(self.operations.capacity(),std::mem::size_of::<ForwardOperation>())?)?;
        for op in &self.operations{
            bytes=add(bytes,op.id.capacity())?;
            bytes=add(bytes,mul(op.inputs.capacity(),std::mem::size_of::<String>())?)?;
            for input in &op.inputs{bytes=add(bytes,input.capacity())?;}
            bytes=add(bytes,mul(op.output.shape.capacity(),std::mem::size_of::<ForwardDimension>())?)?;
            bytes=add(bytes,crate::owned_backing::checked_json_backing_bytes(&op.attributes,max_bytes).map_err(|_|fail())?)?;
        }
        let input_node=crate::owned_backing::checked_tree_node_bytes::<(String,ForwardTensorSpec)>().map_err(|_|fail())?;
        let output_node=crate::owned_backing::checked_tree_node_bytes::<(String,ForwardOutput)>().map_err(|_|fail())?;
        bytes=add(bytes,mul(self.inputs.len().max(1),input_node)?)?;
        bytes=add(bytes,mul(self.outputs.len().max(1),output_node)?)?;
        for (name,spec) in &self.inputs{bytes=add(bytes,name.capacity())?;bytes=add(bytes,mul(spec.shape.capacity(),std::mem::size_of::<ForwardDimension>())?)?;}
        for (name,out) in &self.outputs{bytes=add(bytes,add(name.capacity(),out.value.capacity())?)?;bytes=add(bytes,mul(out.tensor.shape.capacity(),std::mem::size_of::<ForwardDimension>())?)?;}
        if bytes>max_bytes{return Err(fail());}Ok(bytes)
    }

    /// Reconstruct the registered source contract, including exact artifact
    /// names, output augmentation and numeric attributes. A caller-chosen digest
    /// alone is never accepted as a source witness.
    pub fn validate(&self) -> Result<(), ModelError> {
        self.validate_with_limits(crate::default_forward_limits()?)
    }

    /// Carry the caller's admitted metadata policy through reconstruction. The
    /// policy is not serialized into the plan and never substitutes for source
    /// validation or for the private consuming construction owner.
    pub fn validate_with_limits(&self, limits: crate::ForwardLimits) -> Result<(), ModelError> {
        if self.operations.len()>limits.metadata.max_operations_per_phase
            || self.inputs.len() != 4 || !(2..=3).contains(&self.outputs.len()) {
            return Err(ModelError::Incomplete("forward plan collection envelope exceeded".into()));
        }
        for op in &self.operations {
            let valid = match op.operator {
                ModelOperator::AttentionMask => crate::is_key_padding_attention_mask(&op.inputs, &op.attributes),
                ModelOperator::GeluErf => op.inputs.len() == 1 && crate::is_exact_fp32_gelu(&op.attributes),
                ModelOperator::Tanh => op.inputs.len() == 1 && crate::is_fp32_tanh(&op.attributes),
                _ => true,
            };
            if !valid { return Err(ModelError::Incomplete("invalid forward numeric operator contract".into())); }
        }
        let source = self.operations.first().and_then(|op| op.attributes.get("semantic_config"))
            .ok_or_else(|| ModelError::Incomplete("forward source witness missing".into()))?;
        let expected = crate::lower_forward_model_value_with_limits(source, self.max_sequence, limits)?;
        if self != &expected {
            return Err(ModelError::Incomplete("forward plan differs from registered source contract".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardRequirement {
    pub name: String,
    pub dtype: ForwardDType,
    pub shape: Vec<u64>,
}

pub fn forward_checkpoint_requirements(plan: &ForwardPlan) -> Result<Vec<ForwardRequirement>, ModelError> {
    forward_checkpoint_requirements_with_limits(plan,crate::default_forward_limits()?)
}

pub fn forward_checkpoint_requirements_with_limits(plan:&ForwardPlan,limits:crate::ForwardLimits)
    ->Result<Vec<ForwardRequirement>,ModelError>{
    plan.validate_with_limits(limits)?;
    crate::checkpoint_requirements::forward_records(plan)
}

/// Separate native public factory. Not PUMAOriginalSourcePlan and not a release
/// path for protected data. Bounds apply before parsing/expanding graph metadata.
pub fn lower_forward_model(config: &[u8], max_sequence: u64) -> Result<ForwardPlan, ModelError> {
    crate::lower_forward_model_with_limits(config,max_sequence,crate::default_forward_limits()?)
}

pub fn lower_forward_model_value(config: &Value, max_sequence: u64) -> Result<ForwardPlan, ModelError> {
    crate::lower_forward_model_value_with_limits(config,max_sequence,crate::default_forward_limits()?)
}

pub(super) fn invalid(s: &str) -> ModelError { ModelError::InvalidConfig(s.into()) }
#[cfg(test)]
const MAX_SOURCE_BYTES: usize = 1_048_576;

/// Bounded, duplicate-aware model-owned JSON parser. Field spellings selecting
/// exact decimal text are supplied by the registry. The first pass allocates no
/// JSON/String/collection storage; it prepays the subsequent parser tree.
/// Selected numeric fields are validated directly as positive FP32 decimals and
/// retained as original-text Strings. Other numbers keep ordinary Value grammar.
/// Raw parser space is separate from the model Value lifecycle: retained tree/
/// String capacities plus fixed scanner/build workspace. String/key decoding
/// writes directly into a single prepaid exact-capacity String; no serde scratch
/// Vec or copied second destination is live alongside it.
pub fn parse_bounded_source_json(bytes:&[u8],limits:crate::ForwardLimits,exact_decimal_fields:&[&str])
    ->Result<Value,ModelError>{
    use crate::forward_resources::{add,mul};
    if bytes.len()>limits.source_bytes {return Err(invalid("forward source byte limit"));}
    let native=crate::source_work::traversal_limits(limits.source).map_err(|_|invalid("source traversal policy"))?;
    let workspace=add(native.max_workspace_bytes,mul(crate::source_work::SOURCE_FRAME_SLOTS,
        std::mem::size_of::<Lexical<'_>>()+std::mem::size_of::<Value>()+std::mem::size_of::<String>()+
        std::mem::size_of::<Vec<Value>>()+std::mem::size_of::<serde_json::Map<String,Value>>())?)?;
    if workspace>limits.source.max_workspace_bytes || workspace>limits.metadata.max_factory_bytes {
        return Err(invalid("source lexical workspace limit"));
    }
    let work=limits.source.max_total_work;
    if u64::try_from(bytes.len()).map_err(|_|invalid("source size overflow"))?>work{return Err(invalid("source lexical work limit"));}
    if exact_decimal_fields.len()>limits.source.max_nodes {return Err(invalid("exact decimal field limit"));}
    let field_bytes=exact_decimal_fields.iter().try_fold(0,|n,s|add(n,s.len()))?;
    if field_bytes>limits.source.max_payload_bytes {return Err(invalid("exact decimal field payload limit"));}
    std::str::from_utf8(bytes).map_err(|_|invalid("source UTF8"))?;
    let mut scan=Lexical {bytes,at:0,nodes:0,payload:0,backing:std::mem::size_of::<Value>(),work:bytes.len() as u64,
        root_keys:0,root_key_bytes:0,
        ceiling:work,limits,exact:exact_decimal_fields};
    scan.value(1,true,true)?;scan.space();
    if scan.at!=bytes.len() {return Err(invalid("trailing source JSON"));}
    // Include parser storage, root and conservative Vec/String growth before
    // entering construction; node/payload/depth/duplicates already admitted.
    #[cfg(test)]
    PARSER_BUILD_SPACE.with(|state|state.set((scan.backing,workspace)));
    if add(scan.backing,workspace)?>limits.metadata.max_factory_bytes {return Err(invalid("source lexical backing limit"));}
    // BUILD's whitespace is charged once; disjoint string/token spans at most
    // twice their encoded length. Root-field comparisons charge decoded key
    // length + selected-field length exactly as before. Reserve the COMPLETE
    // remainder before the first Map/Vec/String owner can be allocated. This is
    // admission headroom, not an extra meter charge, tariff, or later refund.
    let build_allowance=add(mul(2,bytes.len())?,add(mul(exact_decimal_fields.len(),scan.root_key_bytes)?,
        mul(scan.root_keys,field_bytes)?)?)?;
    let build_allowance=u64::try_from(build_allowance).map_err(|_|invalid("source build work overflow"))?;
    #[cfg(test)]
    PARSER_BUILD_ADMISSION.with(|state|state.set((scan.work,build_allowance)));
    if scan.work.checked_add(build_allowance).ok_or_else(||invalid("source build work overflow"))?>work{
        return Err(invalid("source complete build work limit"));
    }
    scan.at=0;
    scan.build(1,false)
}

#[cfg(test)]
std::thread_local! {
    static PARSER_BUILD_ENTRIES:std::cell::Cell<usize>=const {std::cell::Cell::new(0)};
    static PARSER_BUILD_ADMISSION:std::cell::Cell<(u64,u64)>=const {std::cell::Cell::new((0,0))};
    static PARSER_BUILD_SPACE:std::cell::Cell<(usize,usize)>=const {std::cell::Cell::new((0,0))};
}

struct Lexical<'a> {
    bytes:&'a [u8],at:usize,nodes:usize,payload:usize,backing:usize,work:u64,ceiling:u64,
    root_keys:usize,root_key_bytes:usize,
    limits:crate::ForwardLimits,exact:&'a [&'a str],
}
impl Lexical<'_> {
    fn charge(&mut self,n:usize)->Result<(),ModelError>{
        self.work=self.work.checked_add(u64::try_from(n).map_err(|_|invalid("lexical work overflow"))?)
            .ok_or_else(||invalid("lexical work overflow"))?;
        if self.work>self.ceiling {return Err(invalid("lexical work limit"));} Ok(())
    }
    fn space(&mut self){while self.bytes.get(self.at).is_some_and(|b|matches!(b,b' '|b'\n'|b'\r'|b'\t')){
        if self.work==self.ceiling{break;}self.work+=1;self.at+=1;
    }}
    fn byte(&mut self,b:u8)->Result<(),ModelError>{self.space();if self.bytes.get(self.at)!=Some(&b){return Err(invalid("malformed source JSON"));}self.at+=1;Ok(())}
    fn string(&mut self)->Result<(usize,usize,usize),ModelError>{
        self.space();let start=self.at;self.byte(b'"')?;
        loop {self.charge(1)?;match self.bytes.get(self.at).copied() {
            Some(b'"')=>{self.at+=1;break;},Some(b'\\')=>{self.at+=2;if self.at>self.bytes.len(){return Err(invalid("truncated JSON escape"));}},
            Some(0..=31)|None=>return Err(invalid("malformed JSON string")),Some(_)=>self.at+=1,
        }}
        let end=self.at;
        self.charge(end-start)?;
        let mut decoded=Decoded {bytes:&self.bytes[start+1..end-1],at:0};
        let mut length=0usize;
        while let Some(c)=decoded.next_char()? {length=crate::forward_resources::add(length,c.len_utf8())?;}
        Ok((start,end,length))
    }
    fn owned_string(&mut self)->Result<String,ModelError>{
        let (start,end,length)=self.string()?;
        // Exact decoded UTF8 length was computed allocation-free. Push never
        // grows this destination, including escapes/surrogate pairs. Earlier
        // decoded keys/tree nodes remain covered by retained backing, not by a
        // fixed-workspace claim about an arbitrary-size temporary buffer.
        let mut text=String::with_capacity(length);
        let mut decoded=Decoded {bytes:&self.bytes[start+1..end-1],at:0};
        while let Some(c)=decoded.next_char()?{text.push(c);}
        Ok(text)
    }
    fn equal(&mut self,left:(usize,usize,usize),right:(usize,usize,usize))->Result<bool,ModelError>{
        if left.2!=right.2 {return Ok(false);}
        self.charge(crate::forward_resources::add(left.1-left.0,right.1-right.0)?)?;
        let mut a=Decoded {bytes:&self.bytes[left.0+1..left.1-1],at:0};
        let mut b=Decoded {bytes:&self.bytes[right.0+1..right.1-1],at:0};
        loop{let a=a.next_char()?;let b=b.next_char()?;if a!=b{return Ok(false);}if a.is_none(){return Ok(true);}}
    }
    fn literal_key(&self,key:(usize,usize,usize),literal:&str)->Result<bool,ModelError>{
        let mut a=Decoded {bytes:&self.bytes[key.0+1..key.1-1],at:0};
        for b in literal.chars(){if a.next_char()?!=Some(b){return Ok(false);}} Ok(a.next_char()?.is_none())
    }
    fn text_backing(&mut self,n:usize,key:bool)->Result<(),ModelError>{
        use crate::forward_resources::{add,mul};
        if n>if key {self.limits.source.max_key_bytes}else{self.limits.source.max_string_bytes}{return Err(invalid("source lexical string limit"));}
        self.payload=add(self.payload,n)?;
        if self.payload>self.limits.source.max_payload_bytes{return Err(invalid("source lexical payload limit"));}
        self.backing=add(self.backing,add(mul(2,n)?,8)?)?;Ok(())
    }
    fn token(&mut self)->Result<(usize,usize),ModelError>{
        self.space();let start=self.at;
        while self.bytes.get(self.at).is_some_and(|b|!matches!(b,b' '|b'\n'|b'\r'|b'\t'|b','|b']'|b'}')){self.charge(1)?;self.at+=1;}
        if start==self.at{return Err(invalid("missing JSON scalar"));}self.charge(self.at-start)?;Ok((start,self.at))
    }
    fn value(&mut self,depth:usize,account:bool,duplicates:bool)->Result<(),ModelError>{
        use crate::forward_resources::{add,mul};
        self.space();self.charge(1)?;
        if depth>self.limits.source.max_depth{return Err(invalid("source lexical depth limit"));}
        if account {self.nodes=add(self.nodes,1)?;if self.nodes>self.limits.source.max_nodes{return Err(invalid("source lexical node limit"));}}
        match self.bytes.get(self.at).copied(){
            Some(b'"')=>{let (_,_,len)=self.string()?;if account{self.text_backing(len,false)?;}},
            Some(b'[')=>{
                self.at+=1;let mut count=0;self.space();
                if self.bytes.get(self.at)!=Some(&b']'){loop{self.value(depth+1,account,duplicates)?;count=add(count,1)?;self.space();if self.bytes.get(self.at)==Some(&b']'){break;}self.byte(b',')?;}}
                self.byte(b']')?;
                if account{self.backing=add(self.backing,mul(add(mul(2,count)?,4)?,std::mem::size_of::<Value>())?)?;}
            },
            Some(b'{')=>{
                self.at+=1;let beginning=self.at;let mut count=0;self.space();
                if self.bytes.get(self.at)!=Some(&b'}'){loop{
                    let current=self.string()?;
                    if duplicates{
                        let resume=self.at;self.at=beginning;self.space();
                        while self.at<current.0{
                            let previous=self.string()?;
                            if self.equal(current,previous)?{return Err(invalid("duplicate source key"));}
                            self.byte(b':')?;self.value(depth+1,false,false)?;self.space();
                            if self.bytes.get(self.at)==Some(&b','){self.at+=1;self.space();}else{break;}
                        }
                        self.at=resume;
                    }
                    if account{
                        self.text_backing(current.2,true)?;
                        if depth==1{self.root_keys=add(self.root_keys,1)?;self.root_key_bytes=add(self.root_key_bytes,current.2)?;}
                    }
                    self.byte(b':')?;
                    // Exact-decimal spellings become owned text during build.
                    let mut exact=false;
                    if depth==1{let fields=self.exact;for key in fields{
                        self.charge(crate::forward_resources::add(current.1-current.0,key.len())?)?;
                        exact|=self.literal_key(current,key)?;
                    }}
                    self.space();let numeric=matches!(self.bytes.get(self.at),Some(b'-'|b'0'..=b'9'));
                    let start=self.at;
                    self.value(depth+1,account,duplicates)?;
                    if account&&exact&&numeric{
                        let text=std::str::from_utf8(&self.bytes[start..self.at]).map_err(|_|invalid("decimal encoding"))?;
                        crate::parse_positive_f32(text).map_err(invalid)?;
                        self.text_backing(text.len(),false)?;
                    }
                    count=add(count,1)?;self.space();if self.bytes.get(self.at)==Some(&b'}'){break;}self.byte(b',')?;
                }}self.byte(b'}')?;
                if account{let node=crate::owned_backing::checked_tree_node_bytes::<(String,Value)>().map_err(|_|invalid("source tree policy overflow"))?;
                    self.backing=add(self.backing,mul(count.max(1),node)?)?;}
            },
            Some(_)=>{let (start,end)=self.token()?;
                // Primitive parser only, never a container/string owner. Reject
                // malformed grammar before any source tree can be constructed.
                let scalar:Value=serde_json::from_slice(&self.bytes[start..end]).map_err(|_|invalid("malformed JSON scalar"))?;
                if !matches!(scalar,Value::Null|Value::Bool(_)|Value::Number(_)){return Err(invalid("malformed source scalar"));}
                if account&&matches!(scalar,Value::Number(_)){self.payload=add(self.payload,end-start)?;
                    if self.payload>self.limits.source.max_payload_bytes{return Err(invalid("source lexical payload limit"));}}
            },None=>return Err(invalid("missing source JSON")),
        }Ok(())
    }
    fn build(&mut self,depth:usize,exact:bool)->Result<Value,ModelError>{
        #[cfg(test)]
        PARSER_BUILD_ENTRIES.with(|count|count.set(count.get()+1));
        self.space();match self.bytes.get(self.at).copied(){
            Some(b'"')=>Ok(Value::String(self.owned_string()?)),
            Some(b'[')=>{self.at+=1;let mut values=Vec::new();self.space();if self.bytes.get(self.at)!=Some(&b']'){loop{values.push(self.build(depth+1,false)?);self.space();if self.bytes.get(self.at)==Some(&b']'){break;}self.byte(b',')?;}}self.byte(b']')?;Ok(Value::Array(values))},
            Some(b'{')=>{self.at+=1;let mut values=serde_json::Map::new();self.space();if self.bytes.get(self.at)!=Some(&b'}'){loop{
                let key=self.owned_string()?;
                self.byte(b':')?;
                if depth==1{let fields=self.exact;for field in fields{self.charge(crate::forward_resources::add(key.len(),field.len())?)?;}}
                let decimal=depth==1&&self.exact.contains(&key.as_str());
                let value=self.build(depth+1,decimal)?;values.insert(key,value);self.space();if self.bytes.get(self.at)==Some(&b'}'){break;}self.byte(b',')?;
            }}self.byte(b'}')?;Ok(Value::Object(values))},
            Some(_)=>{let(start,end)=self.token()?;if exact&&matches!(self.bytes[start],b'-'|b'0'..=b'9'){
                Ok(Value::String(std::str::from_utf8(&self.bytes[start..end]).map_err(|_|invalid("decimal encoding"))?.into()))
            }else{serde_json::from_slice(&self.bytes[start..end]).map_err(|_|invalid("source scalar"))}},
            None=>Err(invalid("missing source JSON")),
        }
    }
}

struct Decoded<'a>{bytes:&'a [u8],at:usize}
impl Decoded<'_>{
    fn hex(&mut self)->Result<u32,ModelError>{
        let end=self.at.checked_add(4).ok_or_else(||invalid("JSON unicode overflow"))?;
        let bytes=self.bytes.get(self.at..end).ok_or_else(||invalid("truncated unicode"))?;
        let mut n=0;for b in bytes{n=n*16+match b{b'0'..=b'9'=>(b-b'0') as u32,b'a'..=b'f'=>(b-b'a'+10)as u32,b'A'..=b'F'=>(b-b'A'+10)as u32,_=>return Err(invalid("invalid unicode escape"))};}self.at=end;Ok(n)
    }
    fn next_char(&mut self)->Result<Option<char>,ModelError>{
        if self.at==self.bytes.len(){return Ok(None);}let byte=self.bytes[self.at];self.at+=1;
        if byte!=b'\\'{
            if byte<32{return Err(invalid("unescaped JSON control"));}
            self.at-=1;
            let width=match byte {0..=0x7f=>1,0xc2..=0xdf=>2,0xe0..=0xef=>3,0xf0..=0xf4=>4,_=>return Err(invalid("source UTF8"))};
            let end=self.at.checked_add(width).ok_or_else(||invalid("source UTF8 overflow"))?;
            let text=self.bytes.get(self.at..end).ok_or_else(||invalid("truncated source UTF8"))?;
            let c=std::str::from_utf8(text).map_err(|_|invalid("source UTF8"))?.chars().next().ok_or_else(||invalid("source UTF8"))?;
            self.at+=c.len_utf8();return Ok(Some(c));
        }
        let b=*self.bytes.get(self.at).ok_or_else(||invalid("truncated JSON escape"))?;self.at+=1;
        Ok(Some(match b{b'"'=>'"',b'\\'=>'\\',b'/'=>'/',b'b'=>'\u{8}',b'f'=>'\u{c}',b'n'=>'\n',b'r'=>'\r',b't'=>'\t',b'u'=>{
            let mut n=self.hex()?;if (0xd800..=0xdbff).contains(&n){
                if self.bytes.get(self.at..self.at+2)!=Some(b"\\u".as_slice()){return Err(invalid("unpaired unicode surrogate"));}self.at+=2;
                let low=self.hex()?;if !(0xdc00..=0xdfff).contains(&low){return Err(invalid("unpaired unicode surrogate"));}n=0x10000+(n-0xd800)*1024+(low-0xdc00);
            }char::from_u32(n).ok_or_else(||invalid("invalid unicode scalar"))?
        },_=>return Err(invalid("invalid JSON escape"))}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DecoderWorkload, ModelOperator::*};
    use serde_json::json;
    use sha2::{Digest as _, Sha256};

    // Same genuine source geometry as python/pllm/sources/generated_encoder.py.
    fn source(classifier: bool) -> Value {
        let mut value = json!({
            "model_type":"bert", "architectures":["BertModel"],
            "hidden_size":8, "num_hidden_layers":2, "num_attention_heads":2,
            "intermediate_size":13, "vocab_size":17, "max_position_embeddings":8,
            "type_vocab_size":2, "position_embedding_type":"absolute", "hidden_act":"gelu",
            "layer_norm_eps":1e-12, "hidden_dropout_prob":0.0, "attention_probs_dropout_prob":0.0,
            "classifier_dropout":0.0, "is_decoder":false, "add_cross_attention":false,
            "pad_token_id":0, "torch_dtype":"float32", "initializer_range":0.02,
            "num_labels":3, "problem_type":"single_label_classification",
            "id2label":{"0":"LABEL_0","1":"LABEL_1","2":"LABEL_2"},
            "label2id":{"LABEL_0":0,"LABEL_1":1,"LABEL_2":2}
        });
        if classifier { value["architectures"] = json!(["BertForSequenceClassification"]); }
        value
    }
    fn plan(classifier: bool) -> ForwardPlan { lower_forward_model_value(&source(classifier), 5).unwrap() }
    fn op<'a>(plan: &'a ForwardPlan, id: &str) -> &'a ForwardOperation {
        plan.operations.iter().find(|op| op.id == id).unwrap()
    }

    #[test]
    fn complete_build_work_is_admitted_before_any_build_entry(){
        let bytes=br#"{ "a" : "x", "b" : 2 }"#;
        let base=crate::default_forward_limits().unwrap();
        PARSER_BUILD_ENTRIES.with(|count|count.set(0));
        let expected=parse_bounded_source_json(bytes,base,&["a"]).unwrap();
        assert_eq!(PARSER_BUILD_ENTRIES.with(|count|count.get()),3); // object + two values
        let (scan,allowance)=PARSER_BUILD_ADMISSION.with(|state|state.get());
        assert!(allowance>0);
        // Two one-byte root keys, one one-byte selected-field name.
        assert_eq!(allowance,2*bytes.len()as u64+2+2);
        let mut exact=base;exact.source.max_total_work=scan.checked_add(allowance).unwrap();
        PARSER_BUILD_ENTRIES.with(|count|count.set(0));
        assert_eq!(parse_bounded_source_json(bytes,exact,&["a"]).unwrap(),expected);
        assert_eq!(PARSER_BUILD_ENTRIES.with(|count|count.get()),3);
        exact.source.max_total_work-=1;
        PARSER_BUILD_ENTRIES.with(|count|count.set(0));
        let error=parse_bounded_source_json(bytes,exact,&["a"]).unwrap_err();
        assert!(error.to_string().contains("source complete build work limit"));
        assert_eq!(PARSER_BUILD_ENTRIES.with(|count|count.get()),0);
    }

    #[test]
    fn near_byte_limit_escaped_keys_and_values_have_only_prepaid_destinations(){
        let base=crate::default_forward_limits().unwrap();
        // Decoded length sits above a doubling boundary, while mixed escaped/
        // unescaped bytes fill the byte envelope. An extra geometric scratch
        // Vec plus copied String would exceed the admitted retained+fixed space.
        let n=base.source_bytes*3/5;let escapes=base.source_bytes-8-n;
        let encoded=format!("{}{}","\\n".repeat(escapes),"a".repeat(n-escapes));
        for large_key in [true,false]{
            let bytes=if large_key{format!("{{\"{encoded}\":\"x\"}}")}
                else{format!("{{\"x\":\"{encoded}\"}}")}.into_bytes();
            assert_eq!(bytes.len(),base.source_bytes);
            let parsed=parse_bounded_source_json(&bytes,base,&[]).unwrap();
            let (backing,workspace)=PARSER_BUILD_SPACE.with(|state|state.get());
            let map=parsed.as_object().unwrap();let (key,value)=map.iter().next().unwrap();
            let Value::String(text)=value else{panic!("string value required")};
            assert_eq!(key.len(),if large_key{n}else{1});
            assert_eq!(text.len(),if large_key{1}else{n});
            assert_eq!(key.capacity(),key.len());assert_eq!(text.capacity(),text.len());
            let large=if large_key{key}else{text};
            assert_eq!(large.bytes().filter(|b|*b==b'\n').count(),escapes);
            assert!(n.next_power_of_two()+n>backing+workspace);
            let actual=crate::owned_backing::checked_json_backing_bytes(&parsed,base.metadata.max_factory_bytes).unwrap();
            assert!(actual<=backing); // retained key + current value destination
            let mut exact=base;exact.metadata.max_factory_bytes=backing.checked_add(workspace).unwrap();
            exact.source.max_nodes=2; // object + string; key is not a Value node
            PARSER_BUILD_ENTRIES.with(|count|count.set(0));
            assert!(parse_bounded_source_json(&bytes,exact,&[]).is_ok());
            assert_eq!(PARSER_BUILD_ENTRIES.with(|count|count.get()),2);
            exact.metadata.max_factory_bytes-=1;
            PARSER_BUILD_ENTRIES.with(|count|count.set(0));
            assert!(parse_bounded_source_json(&bytes,exact,&[]).is_err());
            assert_eq!(PARSER_BUILD_ENTRIES.with(|count|count.get()),0);
        }
    }

    #[test]
    fn empty_exact_selection_keeps_ordinary_numbers_and_rejects_nonfinite_before_build(){
        let limits=crate::default_forward_limits().unwrap();
        let value=parse_bounded_source_json(br#"{"a":-1.25,"b":2}"#,limits,&[]).unwrap();
        assert_eq!(value,json!({"a":-1.25,"b":2}));
        assert!(value["a"].is_number());
        for bytes in [b"1e999".as_slice(),b"-1e999",b"NaN",b"Infinity",b"-Infinity"]{
            PARSER_BUILD_ENTRIES.with(|count|count.set(0));
            assert!(parse_bounded_source_json(bytes,limits,&[]).is_err());
            assert_eq!(PARSER_BUILD_ENTRIES.with(|count|count.get()),0);
        }
    }

    #[test]
    fn bytes_and_value_share_registered_factory_and_wire_schema() {
        for classifier in [false, true] {
            let document = source(classifier);
            let p = lower_forward_model_value(&document, 5).unwrap();
            let bytes=serde_json::to_vec(&document).unwrap();
            let parsed=crate::parse_forward_source_json(&bytes,crate::default_forward_limits().unwrap()).unwrap();
            assert_eq!(lower_forward_model(&bytes,5).unwrap(),lower_forward_model_value(&parsed,5).unwrap());
            // The Value API preserves its ORIGINAL numeric witness. The byte
            // parser preserves exact decimal lexemes as strings; these are
            // distinct witnesses, not permission to normalize either digest.
            let (registry,_)=crate::source_mappings::forward_geometry(&document,5).unwrap();
            assert_eq!(p.operations[0].attributes["semantic_config"],document);
            assert_eq!(p.config_digest,canonical_digest(registry.digest_domain,&document));
            p.validate().unwrap();
            assert_eq!(p.inputs.len(), 4);
            assert_eq!(p.operations.len(), if classifier { 50 } else { 49 });
            let wire = serde_json::to_value(&p).unwrap();
            assert_eq!(wire["inputs"]["input.tokens"], json!({"dtype":"int64","shape":[1,"sequence"]}));
            assert_eq!(wire["inputs"]["input.attention_mask"], json!({"dtype":"bool","shape":[1,"sequence"]}));
            for key in ["state", "state_inputs", "state_outputs", "prefill", "decode", "token_feedback", "batch"] {
                assert!(wire.get(key).is_none());
            }
            let replay: ForwardPlan = serde_json::from_value(wire).unwrap();
            assert_eq!(replay, p);
            replay.validate().unwrap();
        }
    }

    #[test]
    fn symbolic_dimensions_and_dtypes_fail_closed() {
        assert_eq!(serde_json::to_value(ForwardDimension::Sequence).unwrap(), json!("sequence"));
        for bad in [json!(0), json!(-1), json!(1.5), json!(null), json!("batch"), json!({})] {
            assert!(serde_json::from_value::<ForwardDimension>(bad).is_err());
        }
        assert!(serde_json::to_value(ForwardDimension::Fixed(0)).is_err());
        assert!(serde_json::from_value::<ForwardDType>(json!("float16")).is_err());
        assert!(serde_json::from_value::<ForwardTensorSpec>(json!({"dtype":"float32","shape":[1],"other":1})).is_err());
    }

    #[test]
    fn exact_41_tensor_classifier_requirements_and_bare_base_names() {
        let p = plan(true);
        let requirements = forward_checkpoint_requirements(&p).unwrap();
        assert_eq!(requirements.len(), 41);
        assert!(requirements.windows(2).all(|r| r[0].name < r[1].name));
        assert!(requirements.iter().all(|r| r.dtype == ForwardDType::Float32));
        let actual: BTreeMap<_, _> = requirements.iter().map(|r| (r.name.clone(), r.shape.clone())).collect();
        let mut expected = BTreeMap::from([
            ("bert.embeddings.word_embeddings.weight".into(), vec![17,8]),
            ("bert.embeddings.position_embeddings.weight".into(), vec![8,8]),
            ("bert.embeddings.token_type_embeddings.weight".into(), vec![2,8]),
        ]);
        let mut pair = |prefix: String, weight: Vec<u64>, width: u64| {
            expected.insert(format!("{prefix}.weight"), weight);
            expected.insert(format!("{prefix}.bias"), vec![width]);
        };
        pair("bert.embeddings.LayerNorm".into(), vec![8], 8);
        for layer in 0..2 {
            let prefix = format!("bert.encoder.layer.{layer}");
            for role in ["query", "key", "value"] { pair(format!("{prefix}.attention.self.{role}"), vec![8,8], 8); }
            pair(format!("{prefix}.attention.output.dense"), vec![8,8], 8);
            pair(format!("{prefix}.attention.output.LayerNorm"), vec![8], 8);
            pair(format!("{prefix}.intermediate.dense"), vec![13,8], 13);
            pair(format!("{prefix}.output.dense"), vec![8,13], 8);
            pair(format!("{prefix}.output.LayerNorm"), vec![8], 8);
        }
        pair("bert.pooler.dense".into(), vec![8,8], 8);
        pair("classifier".into(), vec![3,8], 3);
        assert_eq!(actual, expected);
        let base = forward_checkpoint_requirements(&plan(false)).unwrap();
        assert_eq!(base.len(), 39);
        assert!(base.iter().all(|r| !r.name.starts_with("bert.") && !r.name.starts_with("classifier.")));
        assert!(base.iter().any(|r| r.name == "pooler.dense.bias"));
    }

    #[test]
    fn exact_encoder_numeric_order_and_augmented_outputs() {
        let p = plan(true);
        assert_eq!(op(&p,"embedding.word_segments").inputs, ["embedding.tokens", "embedding.segments"]);
        assert_eq!(op(&p,"embedding.sum").inputs, ["embedding.word_segments", "embedding.positions"]);
        assert_eq!(op(&p,"embedding.positions").attributes["position_policy"], "explicit_arbitrary_valid");
        assert_eq!(op(&p,"embedding.positions").attributes["weight_shape"], json!([8,8]));
        assert_eq!(op(&p,"embedding.segments").attributes["weight_shape"], json!([2,8]));
        assert_eq!(op(&p,"layer.0.query_heads").output.shape,
            vec![ForwardDimension::Fixed(1),ForwardDimension::Fixed(2),ForwardDimension::Sequence,ForwardDimension::Fixed(4)]);
        assert_eq!(op(&p,"layer.0.scaled").attributes, json!({"head_dim":4,"formula":"scores_div_sqrt_head_dim","compute_dtype":"float32"}));
        assert!(crate::is_key_padding_attention_mask(&op(&p,"layer.0.masked").inputs, &op(&p,"layer.0.masked").attributes));
        assert!(crate::is_exact_fp32_gelu(&op(&p,"layer.0.gelu").attributes));
        assert!(crate::is_fp32_tanh(&op(&p,"pooler.output").attributes));
        assert_eq!(op(&p,"pooler.first").attributes, json!({"axis":1,"start":0,"end":1,"squeeze":true}));
        assert_eq!(p.outputs.keys().map(String::as_str).collect::<Vec<_>>(), vec!["last_hidden_state","logits","pooler_output"]);
        assert_eq!(p.outputs["last_hidden_state"].value, "layer.1.output_norm");
        assert_eq!(p.outputs["logits"].tensor.shape, vec![ForwardDimension::Fixed(1),ForwardDimension::Fixed(3)]);
        assert!(!p.operations.iter().any(|op| matches!(op.operator, CausalMask|KvCacheAppend|OutputHead|TokenFeedback|LastToken)));
    }

    #[test]
    fn digest_has_exact_shared_sha256_domain_and_commits_to_sequence() {
        let p = plan(true);
        let mut hash = Sha256::new();
        hash.update(b"pllm.forward_plan.v1\0");
        hash.update(pllm_types::canonical_bytes(&p));
        let digest = Digest::from_sha256(hash.finalize().into());
        assert_eq!(p.digest(), digest);
        assert_ne!(p.digest(), lower_forward_model_value(&source(true), 4).unwrap().digest());
        let mut changed = p.clone();
        changed.config_digest = Digest::from_sha256([0;32]);
        assert!(changed.validate().is_err());
        assert!(forward_checkpoint_requirements(&changed).is_err());
    }

    #[test]
    fn tampered_contracts_artifacts_missing_bias_and_internal_capture_refused() {
        for (id, key, value) in [
            ("layer.0.query", "weight_layout", json!("input_output")),
            ("layer.0.query", "weight", json!("query.weight")),
            ("layer.0.query", "bias", Value::Null),
            ("layer.0.masked", "masked_value", json!("negative_infinity")),
            ("layer.0.masked", "application", json!("replacement")),
            ("layer.0.gelu", "formula", json!("tanh_approximation")),
            ("layer.0.softmax", "compute_dtype", json!("float64")),
            ("embedding.positions", "position_policy", json!("sequential")),
            ("embedding.norm", "epsilon_placement", json!("outside_sqrt")),
        ] {
            let mut p = plan(true);
            p.operations.iter_mut().find(|op| op.id == id).unwrap().attributes[key] = value;
            assert!(p.validate().is_err(), "{id}.{key}");
            assert!(forward_checkpoint_requirements(&p).is_err());
        }
        let mut p = plan(true);
        p.outputs.insert("capture".into(), ForwardOutput { value:"layer.0.gelu".into(), tensor:op(&p,"layer.0.gelu").output.clone() });
        assert!(p.validate().is_err());
        let mut p = plan(false);
        p.operations.retain(|op| op.operator != Tanh);
        assert!(p.validate().is_err());
    }

    #[test]
    fn unsupported_fields_and_malformed_source_contracts_refused() {
        for (key, value) in [
            ("model_type",json!("roberta")), ("architectures",json!(["BertForMaskedLM"])),
            ("architectures",json!(["BertModel","BertForSequenceClassification"])),
            ("is_decoder",json!(true)), ("is_encoder_decoder",Value::Null), ("add_cross_attention",json!(0)),
            ("position_embedding_type",json!("relative_key")), ("hidden_act",json!("gelu_new")),
            ("pruned_heads",json!({"0":[1]})), ("chunk_size_feed_forward",json!(1)),
            ("add_pooling_layer",json!(true)), ("quantization_config",json!({})),
            ("torch_dtype",json!("float16")), ("dtype",json!("float64")),
            ("_attn_implementation",json!("sdpa")), ("output_hidden_states",json!(true)),
            ("output_attentions",json!(true)), ("use_cache",json!(0)), ("return_dict",Value::Null),
            ("hidden_dropout_prob",json!(-0.1)), ("attention_probs_dropout_prob",json!(1.1)),
            ("classifier_dropout",json!("0")), ("pad_token_id",json!(17)), ("initializer_range",json!(-1)),
            ("_name_or_path",json!(0)), ("transformers_version",Value::Null), ("finetuning_task",json!(1)),
            ("problem_type",json!("unknown")), ("hidden_size",json!(7)), ("vocab_size",json!(0)),
            ("num_hidden_layers",json!(u64::MAX)), ("layer_norm_eps",json!("1e-999")),
            ("layer_norm_eps",json!("NaN")), ("num_labels",json!(0)),
            ("id2label",json!({"00":"A","1":"B","2":"C"})),
            ("id2label",json!({"0":"A","1":"A","2":"C"})),
            ("label2id",json!({"LABEL_0":0,"LABEL_1":1,"LABEL_2":1})),
        ] {
            let mut document = source(true);
            document[key] = value;
            assert!(lower_forward_model_value(&document,5).is_err(), "{key}");
            assert!(lower_forward_model(&serde_json::to_vec(&document).unwrap(),5).is_err(), "bytes {key}");
        }
        for key in ["architectures","vocab_size","hidden_size","num_hidden_layers","num_attention_heads",
            "intermediate_size","max_position_embeddings","type_vocab_size","layer_norm_eps","position_embedding_type","hidden_act"] {
            let mut document = source(true);
            document.as_object_mut().unwrap().remove(key);
            assert!(lower_forward_model_value(&document,5).is_err(), "missing {key}");
        }
        assert!(lower_forward_model_value(&source(true),0).is_err());
        assert!(lower_forward_model_value(&source(true),9).is_err());
        assert!(lower_forward_model_value(&source(true),8).is_ok());
    }

    #[test]
    fn legal_eval_metadata_and_explicit_label_resolution() {
        let mut document = source(true);
        for (key,value) in [("pruned_heads",json!({})),("chunk_size_feed_forward",json!(0)),
            ("use_cache",json!(true)),("return_dict",json!(false)),("hidden_dropout_prob",json!(1.0)),
            ("classifier_dropout",Value::Null),("_attn_implementation",Value::Null),("dtype",json!("float32")),
            ("pad_token_id",Value::Null),("finetuning_task",Value::Null),("_name_or_path",json!("local")),
            ("transformers_version",json!("4.57"))] { document[key]=value; }
        let p = lower_forward_model_value(&document,5).unwrap();
        assert_eq!(p.operations.len(),50); // no training dropout operations
        document.as_object_mut().unwrap().remove("num_labels");
        assert!(lower_forward_model_value(&document,5).is_ok()); // explicit map supplies C
        document.as_object_mut().unwrap().remove("id2label");
        document.as_object_mut().unwrap().remove("label2id");
        assert!(lower_forward_model_value(&document,5).is_err()); // never implicit C=2
        document["num_labels"]=json!(4);
        assert_eq!(lower_forward_model_value(&document,5).unwrap().outputs["logits"].tensor.shape,
            vec![ForwardDimension::Fixed(1),ForwardDimension::Fixed(4)]);
    }

    #[test]
    fn epsilon_uses_shared_direct_fp32_parser_not_fp64_roundtrip() {
        let text = "1.0000000596046447753906250000001";
        let mut document = source(false);
        document["layer_norm_eps"] = json!(text);
        let p = lower_forward_model_value(&document,5).unwrap();
        let epsilon = op(&p,"embedding.norm").attributes["epsilon"].as_str().unwrap();
        assert_eq!(crate::parse_positive_f32(epsilon).unwrap().to_bits(),0x3f800001);
        assert_ne!(crate::parse_positive_f32(epsilon).unwrap().to_bits(),(text.parse::<f64>().unwrap() as f32).to_bits());
        assert_eq!(lower_forward_model(&serde_json::to_vec(&document).unwrap(),5).unwrap(),p);
    }

    #[test]
    fn duplicate_keys_trailing_json_and_source_envelopes_refused() {
        for bytes in [br#"{"model_type":"bert","model_type":"bert"}"#.as_slice(),
            br#"{"x":{"a":1,"a":1}}"#.as_slice(),br#"{} {}"#.as_slice()] {
            assert!(lower_forward_model(bytes,1).is_err());
        }
        assert!(lower_forward_model(&vec![b' ';MAX_SOURCE_BYTES+1],1).is_err());
        let mut too_deep=json!(0);
        for _ in 0..18 { too_deep=json!([too_deep]); }
        assert!(lower_forward_model_value(&too_deep,1).is_err());
    }

    #[test]
    fn decoder_identity_and_strict_operator_gate_unchanged() {
        let document = include_bytes!("../tests/fixtures/mini-coder-4b-c87892d-config.json");
        let workload = DecoderWorkload { batch:1,max_input_tokens:2,max_new_tokens:2 };
        let original = crate::lower_model_json(document,workload).unwrap();
        let bytes = pllm_types::canonical_bytes(&original);
        let digest = original.digest();
        let _ = plan(true);
        assert_eq!(pllm_types::canonical_bytes(&crate::lower_model_json(document,workload).unwrap()),bytes);
        assert_eq!(crate::lower_model_json(document,workload).unwrap().digest(),digest);
        for operator in [GeluErf,Tanh,AttentionMask] {
            let mut changed=original.clone();
            changed.prefill.operations[0].operator=operator;
            assert!(changed.validate().is_err());
        }
        for (operator,name) in [(GeluErf,"gelu_erf"),(Tanh,"tanh"),(AttentionMask,"attention_mask")] {
            assert_eq!(serde_json::to_value(operator).unwrap(),json!(name));
        }
    }
}
