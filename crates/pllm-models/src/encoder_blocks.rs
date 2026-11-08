//! Model-neutral post-normalized encoder blocks: learned embedding domains,
//! separate biased attention projections, exact GELU, and pooled readout.
//! Source families, architecture spellings and checkpoint paths live only in
//! the source registry. This builder consumes already validated scalar geometry.
use crate::forward_plan::*;
use crate::source_mappings::EncoderSource;
use crate::{ModelError, ModelOperator};
use pllm_types::canonical_digest;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
pub(super) struct Geometry<'a> {
    pub hidden: u64, pub intermediate: u64, pub layers: u64, pub heads: u64,
    pub vocabulary: u64, pub positions: u64, pub segments: u64,
    pub epsilon: Decimal<'a>, pub labels: Option<u64>,
}

#[derive(Clone, Copy)]
pub(super) enum Decimal<'a> { Text(&'a str), Number(&'a serde_json::Number) }
impl Decimal<'_> {
    pub(super) fn positive_f32(self) -> Result<f32, ModelError> {
        match self {
            Self::Text(s) => crate::parse_positive_f32(s).map_err(invalid),
            Self::Number(n) => {
                // Number's already-owned scalar value, formatted into fixed
                // caller storage. Never clone or allocate epsilon before quote.
                struct Buffer { bytes: [u8; 32], len: usize }
                impl std::fmt::Write for Buffer {
                    fn write_str(&mut self, s: &str) -> std::fmt::Result {
                        let end = self.len.checked_add(s.len()).ok_or(std::fmt::Error)?;
                        if end > self.bytes.len() { return Err(std::fmt::Error); }
                        self.bytes[self.len..end].copy_from_slice(s.as_bytes()); self.len = end; Ok(())
                    }
                }
                use std::fmt::Write;
                let mut buffer = Buffer { bytes: [0;32], len:0 };
                write!(&mut buffer,"{n}").map_err(|_| invalid("epsilon formatting overflow"))?;
                let text = std::str::from_utf8(&buffer.bytes[..buffer.len]).map_err(|_| invalid("epsilon encoding"))?;
                crate::parse_positive_f32(text).map_err(invalid)
            }
        }
    }
}

impl Geometry<'_> {
    pub(super) fn validate(&self, sequence: u64) -> Result<(), ModelError> {
        if self.layers == 0 || self.hidden == 0 || self.intermediate == 0 || self.heads == 0
            || self.hidden % self.heads != 0 || self.vocabulary == 0
            || self.positions == 0 || self.segments == 0
            || sequence == 0 || sequence > self.positions
            || self.labels == Some(0) {
            return Err(invalid("invalid encoder geometry"));
        }
        for dimensions in [[1, 1, sequence, self.hidden], [1, 1, sequence, self.intermediate],
            [1, self.heads, sequence, sequence], [1, 1, self.vocabulary, self.hidden],
            [1, 1, self.intermediate, self.hidden]] {
            dimensions.iter().try_fold(1u64, |n, d| n.checked_mul(*d))
                .ok_or_else(|| invalid("encoder tensor element overflow"))?;
        }
        self.layers.checked_mul(BLOCK_OUTPUTS.len() as u64)
            .and_then(|n| n.checked_add((EMBEDDING_OUTPUTS.len()+POOLER_OUTPUTS.len()) as u64))
            .and_then(|n| n.checked_add(u64::from(self.labels.is_some())))
            .and_then(|n| usize::try_from(n).ok()).ok_or_else(|| invalid("encoder graph expansion overflow"))?;
        self.epsilon.positive_f32()?;
        Ok(())
    }
}

/// Shared output descriptors drive BOTH lowering and numerical quotation. No
/// per-layer descriptor Vec, name formatting or parallel output-total formula.
#[derive(Clone, Copy)]
pub(super) enum OutputShape { Hidden, Heads, Scores, Intermediate, Pooled, Logits }
pub(super) const BLOCK_OUTPUTS: [OutputShape;20] = [
    OutputShape::Hidden, OutputShape::Heads, OutputShape::Hidden, OutputShape::Heads,
    OutputShape::Hidden, OutputShape::Heads, OutputShape::Scores, OutputShape::Scores,
    OutputShape::Scores, OutputShape::Scores, OutputShape::Heads, OutputShape::Hidden,
    OutputShape::Hidden, OutputShape::Hidden, OutputShape::Hidden, OutputShape::Intermediate,
    OutputShape::Intermediate, OutputShape::Hidden, OutputShape::Hidden, OutputShape::Hidden,
];
pub(super) const EMBEDDING_OUTPUTS: [OutputShape;6] = [OutputShape::Hidden;6];
pub(super) const POOLER_OUTPUTS: [OutputShape;3] = [OutputShape::Pooled;3];
pub(super) const BLOCK_IDS:[&str;20]=["query","query_heads","key","key_heads","value","value_heads",
    "scores","scaled","masked","softmax","values","merged",
    "attention_output","attention_residual","attention_norm","intermediate","gelu","output",
    "output_residual","output_norm"];
pub(super) const FORWARD_INPUTS:[(&str,ForwardDType);4]=[("input.tokens",ForwardDType::Int64),
    ("input.positions",ForwardDType::Int64),("input.segments",ForwardDType::Int64),
    ("input.attention_mask",ForwardDType::Bool)];

/// Aggregate exact UTF8 operation-ID bytes without visiting/formating layers.
/// Decimal-index ranges require at most 20 checked scalar steps.
pub(super) fn operation_id_bytes(g:Geometry<'_>)->Result<usize,ModelError>{
    use crate::forward_resources::{add,mul};
    let layers=usize::try_from(g.layers).map_err(|_|invalid("layer count overflow"))?;
    let mut digits=0usize;let mut start=0u64;let mut end=Some(10u64);let mut width=1usize;
    loop{
        let upper=end.unwrap_or(g.layers);
        let count=g.layers.min(upper).checked_sub(start).ok_or_else(||invalid("index range overflow"))?;
        digits=add(digits,mul(usize::try_from(count).map_err(|_|invalid("index count overflow"))?,width)?)?;
        if g.layers<=upper{break;}
        start=upper;width+=1;end=upper.checked_mul(10);
    }
    let fixed=BLOCK_IDS.iter().try_fold(0,|n,role|add(n,add("layer..".len(),role.len())?))?;
    let mut total=add(mul(layers,fixed)?,mul(BLOCK_IDS.len(),digits)?)?;
    for id in ["embedding.tokens","embedding.segments","embedding.positions","embedding.word_segments",
        "embedding.sum","embedding.norm","pooler.first","pooler.linear","pooler.output"] {total=add(total,id.len())?;}
    if g.labels.is_some(){total=add(total,"classifier.logits".len())?;}Ok(total)
}
impl OutputShape {
    pub(super) fn dimensions(self, g: Geometry<'_>, sequence: u64) -> [u64;4] {
        match self {
            Self::Hidden => [1,1,sequence,g.hidden], Self::Heads => [1,g.heads,sequence,g.hidden/g.heads],
            Self::Scores => [1,g.heads,sequence,sequence], Self::Intermediate => [1,1,sequence,g.intermediate],
            Self::Pooled => [1,1,1,g.hidden], Self::Logits => [1,1,1,g.labels.unwrap_or(0)],
        }
    }
    fn tensor(self, g: Geometry<'_>) -> Vec<ForwardDimension> {
        use ForwardDimension::{Fixed as F,Sequence as S};
        match self {
            Self::Hidden => vec![F(1),S,F(g.hidden)], Self::Heads => vec![F(1),F(g.heads),S,F(g.hidden/g.heads)],
            Self::Scores => vec![F(1),F(g.heads),S,S], Self::Intermediate => vec![F(1),S,F(g.intermediate)],
            Self::Pooled => vec![F(1),F(g.hidden)], Self::Logits => vec![F(1),F(g.labels.unwrap_or(0))],
        }
    }
}

fn tensor(shape: Vec<ForwardDimension>) -> ForwardTensorSpec {
    ForwardTensorSpec { dtype: ForwardDType::Float32, shape }
}

struct Builder<'a> {
    source: &'a EncoderSource,
    classifier: bool,
    operations: Vec<ForwardOperation>,
    geometry: Geometry<'a>,
}
impl Builder<'_> {
    fn push(&mut self, id: &str, operator: ModelOperator, layer: Option<u64>, inputs: &[&str],
        shape: Vec<ForwardDimension>, attributes: Value) -> String {
        let output_shape=match layer {
            Some(_) => BLOCK_OUTPUTS[(self.operations.len()-EMBEDDING_OUTPUTS.len())%BLOCK_OUTPUTS.len()].tensor(self.geometry),
            None if self.operations.len()<EMBEDDING_OUTPUTS.len()=>EMBEDDING_OUTPUTS[self.operations.len()].tensor(self.geometry),
            None if operator==ModelOperator::Linear&&id=="classifier.logits"=>OutputShape::Logits.tensor(self.geometry),
            None=>OutputShape::Pooled.tensor(self.geometry),
        };
        let _=shape;
        self.operations.push(ForwardOperation {
            id: id.into(), operator, layer, inputs: inputs.iter().map(|s| (*s).into()).collect(),
            output: tensor(output_shape), attributes,
        });
        id.into()
    }
    fn artifact(&self, layer: Option<u64>, role: &str, bias: bool) -> Result<String, ModelError> {
        self.source.artifact(self.classifier, layer, role, bias)
    }
    fn linear(&mut self, id: &str, input: &str, layer: Option<u64>, role: &str,
        input_width: u64, output_width: u64, shape: Vec<ForwardDimension>) -> Result<String, ModelError> {
        let attrs = json!({"weight":self.artifact(layer, role, false)?,
            "bias":self.artifact(layer, role, true)?, "weight_shape":[output_width,input_width],
            "bias_shape":[output_width], "weight_layout":"output_input",
            "formula":"x_matmul_weight_transpose_plus_bias", "compute_dtype":"float32"});
        Ok(self.push(id, ModelOperator::Linear, layer, &[input], shape, attrs))
    }
    fn norm(&mut self, id: &str, input: &str, layer: Option<u64>, role: &str,
        width: u64, epsilon: &str, shape: Vec<ForwardDimension>) -> Result<String, ModelError> {
        let attrs = json!({"weight":self.artifact(layer, role, false)?,
            "bias":self.artifact(layer, role, true)?, "weight_shape":[width], "bias_shape":[width],
            "axis":-1, "variance":"centered_mean_square", "epsilon":epsilon,
            "epsilon_placement":"inside_sqrt", "affine_order":"normalize_multiply_add", "compute_dtype":"float32"});
        Ok(self.push(id, ModelOperator::LayerNorm, layer, &[input], shape, attrs))
    }
}

pub(super) fn lower<'a>(document: &Value, max_sequence: u64, g: Geometry<'a>, source: &'a EncoderSource)
    -> Result<ForwardPlan, ModelError> {
    use ForwardDimension::{Fixed as F, Sequence as S};
    use ModelOperator::*;
    g.validate(max_sequence)?;
    let classifier = g.labels.is_some();
    let operation_count = usize::try_from(g.layers.checked_mul(20).and_then(|n| n.checked_add(9+u64::from(classifier)))
        .ok_or_else(|| invalid("operation count overflow"))?).map_err(|_| invalid("operation count overflow"))?;
    let mut b = Builder { source, classifier, operations: Vec::with_capacity(operation_count), geometry:g };
    let epsilon = g.epsilon.positive_f32()?.to_string();
    let hidden = vec![F(1), S, F(g.hidden)];
    let split = vec![F(1), F(g.heads), S, F(g.hidden / g.heads)];
    let scores = vec![F(1), F(g.heads), S, S];
    let mut inputs = BTreeMap::new();
    for (name, dtype) in FORWARD_INPUTS {
        inputs.insert(name.into(), ForwardTensorSpec { dtype, shape: vec![F(1), S] });
    }
    for (role, cardinality, domain) in [("tokens", g.vocabulary, "tokens"),
        ("segments", g.segments, "segments"), ("positions", g.positions, "absolute_positions")] {
        let mut attrs = json!({"weight":b.artifact(None, role, false)?,
            "weight_shape":[cardinality,g.hidden], "weight_layout":"output_input",
            "index_domain":domain, "out_of_bounds":"reject", "compute_dtype":"float32"});
        if role == "tokens" { attrs["semantic_config"] = document.clone(); }
        if role == "positions" { attrs["position_policy"] = json!("explicit_arbitrary_valid"); }
        b.push(&format!("embedding.{role}"), TokenLookup, None, &[&format!("input.{role}")], hidden.clone(), attrs);
    }
    b.push("embedding.word_segments", ResidualAdd, None, &["embedding.tokens", "embedding.segments"], hidden.clone(), json!({}));
    b.push("embedding.sum", ResidualAdd, None, &["embedding.word_segments", "embedding.positions"], hidden.clone(), json!({}));
    let mut current = b.norm("embedding.norm", "embedding.sum", None, "embedding_norm", g.hidden, &epsilon, hidden.clone())?;
    for layer in 0..g.layers {
        let name = |role: &str| {
            let role=BLOCK_IDS.iter().copied().find(|id|*id==role).expect("closed encoder block role");
            format!("layer.{layer}.{role}")
        };
        let mut heads = Vec::new();
        for role in ["query", "key", "value"] {
            let projection = b.linear(&name(role), &current, Some(layer), role, g.hidden, g.hidden, hidden.clone())?;
            heads.push(b.push(&name(&format!("{role}_heads")), Reshape, Some(layer), &[&projection], split.clone(),
                json!({"input_layout":"batch_sequence_hidden", "layout":"batch_heads_sequence_feature"})));
        }
        let score = b.push(&name("scores"), AttentionScores, Some(layer), &[&heads[0], &heads[1]], scores.clone(),
            json!({"group_size":1,"compute_dtype":"float32"}));
        let scale = b.push(&name("scaled"), AttentionScale, Some(layer), &[&score], scores.clone(),
            json!({"head_dim":g.hidden/g.heads,"formula":"scores_div_sqrt_head_dim","compute_dtype":"float32"}));
        let mask = b.push(&name("masked"), AttentionMask, Some(layer), &[&scale,"input.attention_mask"], scores.clone(),
            json!({"kind":"key_padding","mask_input":"input.attention_mask","broadcast":"batch_heads_queries_keys",
                "masked_value":"float32_min","application":"additive","compute_dtype":"float32"}));
        let probabilities = b.push(&name("softmax"), Softmax, Some(layer), &[&mask], scores.clone(),
            json!({"axis":-1,"compute_dtype":"float32","formula":"max_subtract_exp_sum"}));
        let values = b.push(&name("values"), AttentionValues, Some(layer), &[&probabilities,&heads[2]], split.clone(),
            json!({"group_size":1,"compute_dtype":"float32"}));
        let merged = b.push(&name("merged"), Reshape, Some(layer), &[&values], hidden.clone(),
            json!({"input_layout":"batch_heads_sequence_feature","layout":"batch_sequence_hidden"}));
        let attention = b.linear(&name("attention_output"), &merged, Some(layer), "attention_output", g.hidden, g.hidden, hidden.clone())?;
        let residual = b.push(&name("attention_residual"), ResidualAdd, Some(layer), &[&current,&attention], hidden.clone(), json!({}));
        let norm = b.norm(&name("attention_norm"), &residual, Some(layer), "attention_norm", g.hidden, &epsilon, hidden.clone())?;
        let inner_shape = vec![F(1), S, F(g.intermediate)];
        let inner = b.linear(&name("intermediate"), &norm, Some(layer), "intermediate", g.hidden, g.intermediate, inner_shape.clone())?;
        let gelu = b.push(&name("gelu"), GeluErf, Some(layer), &[&inner], inner_shape,
            json!({"formula":"x/2*(1+erf(x/sqrt(2)))","compute_dtype":"float32"}));
        let output = b.linear(&name("output"), &gelu, Some(layer), "output", g.intermediate, g.hidden, hidden.clone())?;
        let residual = b.push(&name("output_residual"), ResidualAdd, Some(layer), &[&norm,&output], hidden.clone(), json!({}));
        current = b.norm(&name("output_norm"), &residual, Some(layer), "output_norm", g.hidden, &epsilon, hidden.clone())?;
    }
    let pooled_shape = vec![F(1), F(g.hidden)];
    b.push("pooler.first", Slice, None, &[&current], pooled_shape.clone(), json!({"axis":1,"start":0,"end":1,"squeeze":true}));
    let pooler_linear = b.linear("pooler.linear", "pooler.first", None, "pooler", g.hidden, g.hidden, pooled_shape.clone())?;
    let pooled = b.push("pooler.output", Tanh, None, &[&pooler_linear], pooled_shape.clone(), json!({"compute_dtype":"float32"}));
    let mut outputs = BTreeMap::from([
        ("last_hidden_state".into(), ForwardOutput { value: current, tensor: tensor(hidden) }),
        ("pooler_output".into(), ForwardOutput { value: pooled.clone(), tensor: tensor(pooled_shape) }),
    ]);
    if let Some(labels) = g.labels {
        let shape = vec![F(1), F(labels)];
        let logits = b.linear("classifier.logits", &pooled, None, "classifier", g.hidden, labels, shape.clone())?;
        outputs.insert("logits".into(), ForwardOutput { value: logits, tensor: tensor(shape) });
    }
    Ok(ForwardPlan {
        schema_version: FORWARD_PLAN_SCHEMA_VERSION.into(), model_family: source.family.into(), adapter: source.adapter.into(),
        config_digest: canonical_digest(source.digest_domain, document), max_sequence,
        numeric: FORWARD_NUMERIC.into(), output_contract: FORWARD_OUTPUT_CONTRACT.into(),
        inputs, operations: b.operations, outputs,
    })
}
