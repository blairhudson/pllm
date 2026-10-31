//! UNQUALIFIED synthetic metadata fixtures. No issuer, encoding, native call or
//! production-admission bypass. Direct emission returns only `Unqualified` here;
//! it CANNOT construct `Checked` or be used as a production admission token.
use super::*;
use pllm_core::row_fold::{Geometry, WorkLedger};

const RELEASES:[&[usize];18]=[
    &[],&[],&[2],&[],&[4],&[3,5],&[6],&[1],&[8,7],&[9],&[10],
    &[],&[12],&[],&[14],&[13,15],&[11,16],&[17],
];
// Hand-authored contracts. Neither source cursor nor adapters produce expectations.
fn unary(selector:Selector,parent:usize)->Instruction { Instruction{selector,parents:[parent,NONE],arity:1} }
fn binary(kind:Arithmetic,left:usize,right:usize)->Instruction {
    Instruction{selector:Selector::Arithmetic(kind),parents:[left,right],arity:2}
}
fn row_expected()->[Instruction;18] {
    use Arithmetic::*;
    use Selector::{RowVector as P,RowRotation as R,Arithmetic as A};
    [unary(P(0),0),unary(R(1),1),unary(P(2),2),unary(R(-1),1),unary(P(3),4),
     binary(Add,3,5),binary(Add,1,6),unary(P(1),1),binary(Subtract,8,7),unary(P(0),9),
     unary(A(Square),10),unary(R(1),11),unary(P(2),12),unary(R(-1),11),unary(P(3),14),
     binary(Add,13,15),binary(Add,11,16),unary(P(0),17)]
}
fn pc_expected()->[Instruction;88] {
    use super::Arithmetic::Add;
    use Branch::*;
    use Selector::*;
    [
     unary(Weight{q:0,a:0,j:0},0), unary(Weight{q:0,a:0,j:1},1), binary(Add,4,5),
     unary(Drop{q:0,a:0,boundary:1,branch:None},6), unary(Mask{q:0,a:0,branch:Support},7),
     unary(Drop{q:0,a:0,boundary:2,branch:Some(Support)},8), binary(Add,2,9),
     unary(Weight{q:0,a:1,j:0},0), unary(Weight{q:0,a:1,j:1},1), binary(Add,11,12),
     unary(Drop{q:0,a:1,boundary:1,branch:None},13),
     unary(Mask{q:0,a:1,branch:Wrap},14),
     unary(Drop{q:0,a:1,boundary:2,branch:Some(Wrap)},15),
     unary(Mask{q:0,a:1,branch:Remainder},14),
     unary(Drop{q:0,a:1,boundary:2,branch:Some(Remainder)},17),
     unary(Rotation{q:0,a:1,shift:3},16), unary(Rotation{q:0,a:1,shift:-1},18),
     binary(Add,19,20), binary(Add,10,21),
     unary(Weight{q:0,a:2,j:0},0), unary(Weight{q:0,a:2,j:1},1), binary(Add,23,24),
     unary(Drop{q:0,a:2,boundary:1,branch:None},25),
     unary(Mask{q:0,a:2,branch:Wrap},26),
     unary(Drop{q:0,a:2,boundary:2,branch:Some(Wrap)},27),
     unary(Mask{q:0,a:2,branch:Remainder},26),
     unary(Drop{q:0,a:2,boundary:2,branch:Some(Remainder)},29),
     unary(Rotation{q:0,a:2,shift:2},28), unary(Rotation{q:0,a:2,shift:-2},30),
     binary(Add,31,32), binary(Add,22,33),
     unary(Weight{q:0,a:3,j:0},0), unary(Weight{q:0,a:3,j:1},1), binary(Add,35,36),
     unary(Drop{q:0,a:3,boundary:1,branch:None},37),
     unary(Mask{q:0,a:3,branch:Wrap},38),
     unary(Drop{q:0,a:3,boundary:2,branch:Some(Wrap)},39),
     unary(Mask{q:0,a:3,branch:Remainder},38),
     unary(Drop{q:0,a:3,boundary:2,branch:Some(Remainder)},41),
     unary(Rotation{q:0,a:3,shift:1},40), unary(Rotation{q:0,a:3,shift:-3},42),
     binary(Add,43,44), binary(Add,34,45),
     unary(Weight{q:1,a:0,j:0},0), unary(Weight{q:1,a:0,j:1},1), binary(Add,47,48),
     unary(Drop{q:1,a:0,boundary:1,branch:None},49), unary(Mask{q:1,a:0,branch:Support},50),
     unary(Drop{q:1,a:0,boundary:2,branch:Some(Support)},51), binary(Add,3,52),
     unary(Weight{q:1,a:1,j:0},0), unary(Weight{q:1,a:1,j:1},1), binary(Add,54,55),
     unary(Drop{q:1,a:1,boundary:1,branch:None},56),
     unary(Mask{q:1,a:1,branch:Wrap},57),
     unary(Drop{q:1,a:1,boundary:2,branch:Some(Wrap)},58),
     unary(Mask{q:1,a:1,branch:Remainder},57),
     unary(Drop{q:1,a:1,boundary:2,branch:Some(Remainder)},60),
     unary(Rotation{q:1,a:1,shift:3},59), unary(Rotation{q:1,a:1,shift:-1},61),
     binary(Add,62,63), binary(Add,53,64),
     unary(Weight{q:1,a:2,j:0},0), unary(Weight{q:1,a:2,j:1},1), binary(Add,66,67),
     unary(Drop{q:1,a:2,boundary:1,branch:None},68),
     unary(Mask{q:1,a:2,branch:Wrap},69),
     unary(Drop{q:1,a:2,boundary:2,branch:Some(Wrap)},70),
     unary(Mask{q:1,a:2,branch:Remainder},69),
     unary(Drop{q:1,a:2,boundary:2,branch:Some(Remainder)},72),
     unary(Rotation{q:1,a:2,shift:2},71), unary(Rotation{q:1,a:2,shift:-2},73),
     binary(Add,74,75), binary(Add,65,76),
     unary(Weight{q:1,a:3,j:0},0), unary(Weight{q:1,a:3,j:1},1), binary(Add,78,79),
     unary(Drop{q:1,a:3,boundary:1,branch:None},80),
     unary(Mask{q:1,a:3,branch:Wrap},81),
     unary(Drop{q:1,a:3,boundary:2,branch:Some(Wrap)},82),
     unary(Mask{q:1,a:3,branch:Remainder},81),
     unary(Drop{q:1,a:3,boundary:2,branch:Some(Remainder)},84),
     unary(Rotation{q:1,a:3,shift:1},83), unary(Rotation{q:1,a:3,shift:-3},85),
     binary(Add,86,87), binary(Add,77,88), unary(Bias(0),46), unary(Bias(1),89),
    ]
}
struct Unqualified<'a> { storage: &'a Storage }
fn inspect_unqualified<'a>(source:&Source<'_>,b:&Bindings<'_>,storage:&'a mut Storage)
    ->Result<Unqualified<'a>,Refusal> {
    let tables=preflight(source,b,storage,source.count()?)?;
    fill(source,b,storage,&tables);
    Ok(Unqualified{storage})
}
fn state()->PublicCiphertextState {
    PublicCiphertextState{scheme:"synthetic".into(),key_domain:"unqualified".into(),
        layout:"two_slots".into(),scale:"external_test_only".into(),level:3,shape:vec![2],
        error:PublicCiphertextError{model_identity:"test_error".into(),bound:Some(0)}}
}
fn cap(index:usize,arity:usize)->PublicCiphertextCapability {
    let s=state();
    PublicCiphertextCapability{identity:format!("c{index:02}"),kernel_identity:format!("k{index:02}"),
        primitive:PublicPrimitive::Arithmetic,inputs:vec![s.clone();arity],output:s,
        join:if arity==1{PublicCiphertextJoin::Unary}else{PublicCiphertextJoin::Aligned},
        execution:PublicCiphertextExecution::Ckks{
            compute:PublicLevelCompute{input_level:3,cost:PublicCost{identity:"compute".into(),units:Some(1)}},
            depth:0,transition:PublicLevelTransition{kind:PublicLevelTransitionKind::Preserve,
                from_level:3,to_level:3,cost:PublicCost{identity:"preserve".into(),units:Some(0)}}},
        error_assumption:Some("UNQUALIFIED_metadata_only".into()),scratch_bytes:Some(0),recovery_overlap_bytes:Some(0)}
}
struct Synthetic<'a> {
    values:Vec<PublicCiphertextValue>,caps:Vec<PublicCiphertextCapability>,rows:Vec<Binding<'a>>,storage:Storage,
}
fn synthetic<'a>(_plan:&InstructionPlan,public:&'a Vec<f64>,encoded:&'a OriginalOwner<'a>)->Synthetic<'a> {
    let values=(0..19).map(|i|PublicCiphertextValue{identity:format!("v{i:02}"),owner:format!("ct{i:02}"),state:state()}).collect::<Vec<_>>();
    let mut caps=Vec::new(); let mut rows=Vec::new(); let mut operations=Vec::new(); let mut trace=Vec::new();
    for i in 0..18 {
        let ins=row_expected()[i];
        if !rows.iter().any(|r:&Binding<'_>|r.selector==ins.selector) {
            let index=caps.len(); caps.push(cap(index,ins.arity));
            let evidence=match ins.selector {
                Selector::RowVector(vector)=>Evidence::Operand{source:Operand::RowVector{source:public,vector},encoded},
                Selector::RowRotation(signed_offset)=>Evidence::Rotation{signed_offset,key:encoded,key_domain:"unqualified",layout:"two_slots"},
                Selector::Arithmetic(_)=>Evidence::Arithmetic,_=>panic!("row fixture only"),
            };
            rows.push(Binding{selector:ins.selector,capability:index,evidence});
        }
        operations.push(PublicCiphertextGraphOperation{identity:format!("o{i:02}"),kernel_identity:"xxx".into(),
            primitive:PublicPrimitive::Arithmetic,inputs:vec!["xxx".into();ins.arity],
            output:PublicCiphertextValue{identity:"xxx".into(),owner:"xxxx".into(),state:state()}});
        trace.push(PublicCiphertextTraceStep{operation:"xxx".into(),capability:"xxx".into(),release:vec!["xxx".into();RELEASES[i].len()]});
    }
    let owners=values.iter().map(|v|PublicCiphertextOwner{identity:v.owner.clone(),bytes:Some(64)}).chain(std::iter::once(encoded.owner.clone())).collect();
    let storage=Storage::new(PublicCiphertextGraph{cost_model_identity:"test".into(),cost_unit:"unit".into(),maximum_level:3,
        owners,persistent_owners:vec![encoded.owner.identity.clone()],inputs:vec![values[0].clone()],operations,
        capabilities:caps.clone(),outputs:vec![values[18].identity.clone()],retained_inputs:vec![values[0].identity.clone()]},trace);
    Synthetic{values,caps,rows,storage}
}
fn bindings<'a>(f:&'a Synthetic<'a>,owners:&'a Vec<OriginalOwner<'a>>,budget:&'a Budget,guard:&'a ExternalCharge<'a>)->Bindings<'a> {
    Bindings{rows:&f.rows,capabilities:&f.caps,values:&f.values,owners,zeros:None,budget,overlap:None,
        metadata:MetadataCharge{guard,original_bytes:None,manifest_other:0}}
}

#[test]
fn unqualified_direct_emission_inspects_real_ssa_and_literal_releases_without_admission() {
    let budget=Budget::new(1<<20).unwrap(); let guard=budget.charge_external(64).unwrap();
    let raw=vec![0u8;64]; let owner=PublicCiphertextOwner{identity:"encoded".into(),bytes:Some(64)};
    let encoded_state=state();
    let owners=vec![OriginalOwner{owner:&owner,backing:OriginalBacking::Bytes(&raw),original_bytes:Some(64),guard:&guard,manifest_other:0,parameter_state:Some(&encoded_state)}];
    let mut work=WorkLedger::new(65536,65536);
    let plan=InstructionPlan::new(Geometry::new(1,2,1).unwrap(),&mut work).unwrap();
    let public=vec![f64::NAN;8]; // Never scanned or filled by source lowering.
    let source=Source::Row{plan:&plan,public:&public,original:&owners[0]};
    let mut f=synthetic(&plan,&public,&owners[0]);
    let b=Bindings{rows:&f.rows,capabilities:&f.caps,values:&f.values,owners:&owners,zeros:None,budget:&budget,overlap:None,
        metadata:MetadataCharge{guard:&guard,original_bytes:None,manifest_other:0}};
    let unqualified=inspect_unqualified(&source,&b,&mut f.storage).unwrap();
    for (i,(op,step)) in unqualified.storage.graph.operations.iter().zip(&unqualified.storage.trace).enumerate() {
        let original=row_expected()[i];
        assert_eq!(op.primitive,PublicPrimitive::Arithmetic);
        assert_eq!(op.output,f.values[i+1]);
        assert_eq!(op.inputs,original.parents[..original.arity].iter().map(|&p|f.values[p].identity.clone()).collect::<Vec<_>>());
        assert_eq!(step.operation,op.identity);
        assert_eq!(step.release,RELEASES[i].iter().map(|&o|f.values[o].identity.clone()).collect::<Vec<_>>());
        let cap=&f.caps[f.rows.iter().find(|r|r.selector==original.selector).unwrap().capability];
        assert_eq!(op.kernel_identity,cap.kernel_identity); assert_eq!(step.capability,cap.identity);
        assert_eq!(op.output.state,cap.output);
        assert_eq!(cap.inputs,original.parents[..original.arity].iter().map(|&p|f.values[p].state.clone()).collect::<Vec<_>>());
    }
    assert_eq!(unqualified.storage.graph.retained_inputs,["v00"]);
    assert_eq!(unqualified.storage.graph.outputs,["v18"]);
    assert!(unqualified.storage.poisoned);
    assert!(public.iter().all(|v|v.is_nan()));
    // Same selector is reusable, different selectors may not share bindings.
    assert_eq!(row_expected()[0].selector,row_expected()[9].selector);
    for (i,row) in f.rows.iter().enumerate() { for other in &f.rows[..i] {
        if row.selector!=other.selector {
            assert_ne!(row.capability,other.capability);
            assert_ne!(f.caps[row.capability].kernel_identity,f.caps[other.capability].kernel_identity);
        }
    }}
    let mut swapped=row_expected()[8];swapped.parents.swap(0,1);
    assert_ne!(unqualified.storage.graph.operations[8].inputs,
        swapped.parents[..swapped.arity].iter().map(|&p|f.values[p].identity.clone()).collect::<Vec<_>>());
}

#[test]
fn missing_rows_wrong_key_sign_and_source_vector_refuse_before_writes() {
    let budget=Budget::new(1<<20).unwrap(); let guard=budget.charge_external(64).unwrap();
    let raw=vec![0u8;64]; let owner=PublicCiphertextOwner{identity:"encoded".into(),bytes:Some(64)};
    let encoded_state=state();
    let owners=vec![OriginalOwner{owner:&owner,backing:OriginalBacking::Bytes(&raw),original_bytes:Some(64),guard:&guard,manifest_other:0,parameter_state:Some(&encoded_state)}];
    let mut work=WorkLedger::new(65536,65536); let plan=InstructionPlan::new(Geometry::new(1,2,1).unwrap(),&mut work).unwrap();
    let public=vec![0.;8]; let source=Source::Row{plan:&plan,public:&public,original:&owners[0]};
    for case in 0..4 {
        let mut f=synthetic(&plan,&public,&owners[0]);
        match case {
            0=>{f.rows.remove(0);},
            1=>{let r=f.rows.iter_mut().find(|r|r.selector==Selector::RowRotation(1)).unwrap();
                if let Evidence::Rotation{signed_offset,..}=&mut r.evidence {*signed_offset=-1;}},
            2=>{if let Evidence::Operand{source:Operand::RowVector{vector,..},..}=&mut f.rows[0].evidence {*vector=1;}},
            _=>{f.rows[0].capability=usize::MAX;},
        }
        let b=bindings(&f,&owners,&budget,&guard); let before=f.storage.graph.clone(); let trace=f.storage.trace.clone();
        assert!(preflight(&source,&b,&f.storage,18).is_err());
        assert_eq!(f.storage.graph,before); assert_eq!(f.storage.trace,trace); assert!(!f.storage.poisoned);
    }
}

#[test]
fn foreign_state_alias_short_nested_slots_and_missing_cost_refuse_without_publication() {
    let budget=Budget::new(1<<20).unwrap(); let guard=budget.charge_external(64).unwrap();
    let raw=vec![0u8;64]; let owner=PublicCiphertextOwner{identity:"encoded".into(),bytes:Some(64)};
    let encoded_state=state();
    let owners=vec![OriginalOwner{owner:&owner,backing:OriginalBacking::Bytes(&raw),original_bytes:Some(64),guard:&guard,manifest_other:0,parameter_state:Some(&encoded_state)}];
    let mut work=WorkLedger::new(65536,65536); let plan=InstructionPlan::new(Geometry::new(1,2,1).unwrap(),&mut work).unwrap();
    let public=vec![0.;8]; let source=Source::Row{plan:&plan,public:&public,original:&owners[0]};
    for case in 0..10 {
        let mut f=synthetic(&plan,&public,&owners[0]);
        match case {
            0=>f.values[1].state.key_domain="foreign".into(),
            1=>f.values[1].state.scale="foreign".into(),
            2=>f.values[1].state.error.bound=None,
            3=>f.values[1].owner=f.values[0].owner.clone(),
            4=>f.storage.graph.operations[0].inputs[0]=String::new(),
            5=>f.storage.trace[8].release.pop().map(|_|()).unwrap(),
            6=>f.storage.graph.operations[0].output.state.shape=Vec::new(),
            7=>{if let PublicCiphertextExecution::Ckks{compute,..}=&mut f.caps[0].execution {compute.cost.units=None;}
                f.storage.graph.capabilities=f.caps.clone();},
            8=>f.values[1].state.error.model_identity="foreign".into(),
            _=>{if let PublicCiphertextExecution::Ckks{transition,..}=&mut f.caps[0].execution {transition.cost.units=Some(1);}
                f.storage.graph.capabilities=f.caps.clone();},
        }
        let b=bindings(&f,&owners,&budget,&guard); let before=f.storage.graph.clone(); let trace=f.storage.trace.clone();
        assert!(preflight(&source,&b,&f.storage,18).is_err(),"case {case}");
        assert_eq!(f.storage.graph,before); assert_eq!(f.storage.trace,trace); assert!(!f.storage.poisoned);
    }
}

#[test]
fn failed_subtraction_does_not_spend_and_successful_debit_never_refunds() {
    let mut work=WorkLedger::new(10,10);
    assert!(work.debit(0,11).is_err());assert_eq!(work.remaining(),10);
    work.debit(0,6).unwrap();assert_eq!(work.remaining(),4);
    let _=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||panic!("after prepayment")));
    assert_eq!(work.remaining(),4);assert!(work.debit(0,6).is_err());assert_eq!(work.remaining(),4);
    let mut metadata=MetadataLedger::new(65536);
    assert!(metadata.prepay_work(65537).is_err());assert_eq!(metadata.remaining(),65536);
    metadata.prepay_work(65536).unwrap();assert_eq!(metadata.remaining(),0);
    assert!(metadata.prepay_work(1).is_err());assert_eq!(metadata.remaining(),0);
}

#[test]
fn resource_formula_is_requirement_never_an_ambient_grant() {
    let records=7774usize;
    let required=64*records*records*records+256*records;
    assert_eq!(required,30_068_627_962_880);
    let mut metadata=MetadataLedger::new(65536);
    assert_eq!(metadata.prepay_work(required),Err(construction::Refusal::MetadataWorkRefused));
    assert_eq!(metadata.remaining(),65536);
    assert_eq!(add(usize::MAX,1),Err(Refusal::Overflow));
    assert_eq!(mul(usize::MAX,2),Err(Refusal::Overflow));
}

#[test]
fn original_full_backing_capacity_not_visible_prefix_is_required() {
    let mut bytes=Vec::with_capacity(32);bytes.extend_from_slice(&[1u8,2,3]);
    let backing=OriginalBacking::Bytes(&bytes);
    assert_eq!(backing.range(bytes.capacity()-1),Err(Refusal::Capacity));
    let (start,end)=backing.range(bytes.capacity()).unwrap();
    assert_eq!(end-start,bytes.capacity());assert_eq!(bytes.len(),3);
    assert_eq!(backing.range(usize::MAX),Err(Refusal::Overflow));
}

#[test]
fn final_identity_categories_refuse_before_private_writes() {
    let budget=Budget::new(1<<20).unwrap();let guard=budget.charge_external(64).unwrap();
    let raw=vec![0u8;64];let owner=PublicCiphertextOwner{identity:"encoded".into(),bytes:Some(64)};
    let encoded_state=state();
    let owners=vec![OriginalOwner{owner:&owner,backing:OriginalBacking::Bytes(&raw),original_bytes:Some(64),
        guard:&guard,manifest_other:0,parameter_state:Some(&encoded_state)}];
    let mut work=WorkLedger::new(65536,65536);let plan=InstructionPlan::new(Geometry::new(1,2,1).unwrap(),&mut work).unwrap();
    let public=vec![0.;8];let source=Source::Row{plan:&plan,public:&public,original:&owners[0]};
    for bad in [String::new(),"bad id".into(),"é".into(),"a".repeat(65)] {
        assert!(!bounded(&bad));
        for category in 0..13 {
            let mut f=synthetic(&plan,&public,&owners[0]);
            match category {
                0=>f.storage.graph.cost_model_identity=bad.clone(),1=>f.storage.graph.cost_unit=bad.clone(),
                2=>f.values[1].identity=bad.clone(),3=>f.values[1].owner=bad.clone(),
                4=>f.values[1].state.scheme=bad.clone(),5=>f.values[1].state.error.model_identity=bad.clone(),
                6=>f.storage.graph.operations[0].identity=bad.clone(),
                7=>f.caps[0].identity=bad.clone(),8=>f.caps[0].kernel_identity=bad.clone(),
                9=>f.caps[0].error_assumption=Some(bad.clone()),
                10=>{if let PublicCiphertextExecution::Ckks{compute,..}=&mut f.caps[0].execution {compute.cost.identity=bad.clone();}},
                11=>f.storage.graph.outputs[0]=bad.clone(),_=>f.storage.graph.owners[0].identity=bad.clone(),
            }
            f.storage.graph.capabilities=f.caps.clone();
            let b=bindings(&f,&owners,&budget,&guard);let before=f.storage.graph.clone();let trace=f.storage.trace.clone();
            assert!(preflight(&source,&b,&f.storage,18).is_err(),"identity category {category}");
            assert_eq!(f.storage.graph,before);assert_eq!(f.storage.trace,trace);assert!(!f.storage.poisoned);
        }
    }
    assert!(bounded("valid./_:-ID09"));assert!(bounded(&"a".repeat(64)));
    let mut f=synthetic(&plan,&public,&owners[0]);
    f.storage.graph.cost_model_identity="valid./_:-ID09".into();f.storage.graph.cost_unit="unit./_:-".into();
    let b=bindings(&f,&owners,&budget,&guard);assert!(preflight(&source,&b,&f.storage,18).is_ok());
}

#[test]
fn declared_custodied_but_unused_owner_refuses_persistent_only_owner_accepts() {
    let budget=Budget::new(1<<20).unwrap();let guard=budget.charge_external(64).unwrap();
    let extra_guard=budget.charge_external(64).unwrap();let raw=vec![0u8;64];let extra_raw=vec![0u8;64];
    let owner=PublicCiphertextOwner{identity:"encoded".into(),bytes:Some(64)};
    let extra=PublicCiphertextOwner{identity:"extra".into(),bytes:Some(64)};let encoded_state=state();
    let owners=vec![OriginalOwner{owner:&owner,backing:OriginalBacking::Bytes(&raw),original_bytes:Some(64),guard:&guard,
        manifest_other:0,parameter_state:Some(&encoded_state)},OriginalOwner{owner:&extra,backing:OriginalBacking::Bytes(&extra_raw),
        original_bytes:Some(64),guard:&extra_guard,manifest_other:1,parameter_state:None}];
    assert!(owners[1].guard.is_for(&budget));assert_eq!(owners[1].guard.payload_bytes(),64);
    assert!(owners[1].backing.range(64).is_ok());
    let mut work=WorkLedger::new(65536,65536);let plan=InstructionPlan::new(Geometry::new(1,2,1).unwrap(),&mut work).unwrap();
    let public=vec![0.;8];let source=Source::Row{plan:&plan,public:&public,original:&owners[0]};
    let mut f=synthetic(&plan,&public,&owners[0]);f.storage.graph.owners.push(extra.clone());
    let before=f.storage.graph.clone();let trace=f.storage.trace.clone();
    {let b=bindings(&f,&owners,&budget,&guard);assert!(matches!(preflight(&source,&b,&f.storage,18),Err(Refusal::Binding)));}
    assert_eq!(f.storage.graph,before);assert_eq!(f.storage.trace,trace);assert!(!f.storage.poisoned);
    f.storage.graph.persistent_owners.push(extra.identity.clone());
    let b=bindings(&f,&owners,&budget,&guard);assert!(preflight(&source,&b,&f.storage,18).is_ok());
}

#[test]
fn local_final_control_guard_boundaries_coupled_to_real_construction_debit() {
    // Local construction/guard coverage only, NOT successful full lower admission.
    // Fixed small ambient grants chosen independently of measured requirement.
    for case in 0..3 {
        let graph=PublicCiphertextGraph{cost_model_identity:"test".into(),cost_unit:"unit".into(),maximum_level:3,
            owners:Vec::new(),persistent_owners:Vec::new(),inputs:Vec::new(),operations:Vec::new(),capabilities:Vec::new(),
            outputs:Vec::new(),retained_inputs:Vec::new()};
        let storage=Storage::new(graph,Vec::new());
        let mut backings=Vec::new();
        for s in [&storage.graph.cost_model_identity,&storage.graph.cost_unit] {
            backings.push(construction::Backing{element:construction::Element::Byte,length:s.len(),capacity:Some(s.capacity()),first_phase:0,end_phase:1});
        }
        for element in [construction::Element::Owner,construction::Element::String,construction::Element::Value,
            construction::Element::Operation,construction::Element::Capability,construction::Element::String,
            construction::Element::String,construction::Element::Step] {
            backings.push(construction::Backing{element,length:0,capacity:Some(0),first_phase:0,end_phase:1});
        }
        let manifest=PublicCiphertextConstructionManifest{owners:0,persistent_owners:0,input_ranks:&[],operations:&[],capabilities:&[],
            outputs:0,retained_inputs:0,release_lengths:&[],backings:&backings,other_owners:&[],phases:1,manifest_original_bytes:Some(65536)};
        let mut work=WorkLedger::new(131072,65536);let mut control=ControlLedger::new(&mut work);
        let quote=construction::quote_public_ciphertext_dag_construction(&manifest,&mut control).unwrap();
        let cost=final_control_cost(&manifest).unwrap();
        let before=(control.core_work().remaining(),control.core_work().control_remaining());
        quote.check_constructed(&storage.graph,&storage.trace,&mut control).unwrap();
        assert_eq!((control.core_work().remaining(),control.core_work().control_remaining()),(before.0-cost,before.1-cost));
        let target=(if case==0{cost-1}else{cost},if case==1{cost-1}else{cost});
        let spend=control.core_work().control_remaining()-target.1;control.core_work().debit(0,spend).unwrap();
        let spend=control.core_work().remaining()-target.0;control.core_work().debit(spend,0).unwrap();
        let metadata=MetadataLedger::new(65536);let graph=storage.graph.clone();let trace=storage.trace.clone();
        for _ in 0..2 {
            let result=guard_final_control(&manifest,&mut control);
            if case==2 {assert_eq!(result,Ok(()));}else{
                assert_eq!(result,Err(Refusal::Construction(construction::Refusal::ControlWorkExhausted)));
            }
            assert_eq!((control.core_work().remaining(),control.core_work().control_remaining()),target);
            assert_eq!(metadata.remaining(),65536);assert_eq!(storage.graph,graph);assert_eq!(storage.trace,trace);assert!(!storage.poisoned);
        }
        if case==2 {
            quote.check_constructed(&storage.graph,&storage.trace,&mut control).unwrap();
            assert_eq!((control.core_work().remaining(),control.core_work().control_remaining()),(0,0));
            assert_eq!(guard_final_control(&manifest,&mut control),Err(Refusal::Construction(construction::Refusal::ControlWorkExhausted)));
            assert_eq!(quote.check_constructed(&storage.graph,&storage.trace,&mut control),Err(construction::Refusal::ControlWorkExhausted));
            assert_eq!((control.core_work().remaining(),control.core_work().control_remaining()),(0,0));
        }
    }
}

#[test]
fn small_original_grant_refuses_lower_before_writes_retries_do_not_refund() {
    let budget=Budget::new(1<<20).unwrap();let guard=budget.charge_external(64).unwrap();
    let raw=vec![0u8;64];let owner=PublicCiphertextOwner{identity:"encoded".into(),bytes:Some(64)};
    let encoded_state=state();
    let owners=vec![OriginalOwner{owner:&owner,backing:OriginalBacking::Bytes(&raw),original_bytes:Some(64),
        guard:&guard,manifest_other:0,parameter_state:Some(&encoded_state)}];
    let mut work=WorkLedger::new(65536,65536);
    let plan=InstructionPlan::new(Geometry::new(1,2,1).unwrap(),&mut work).unwrap();
    let public=vec![f64::NAN;8];let mut f=synthetic(&plan,&public,&owners[0]);
    let b=Bindings{rows:&f.rows,capabilities:&f.caps,values:&f.values,owners:&owners,zeros:None,budget:&budget,overlap:None,
        metadata:MetadataCharge{guard:&guard,original_bytes:None,manifest_other:0}};
    // Deliberately incomplete original inventory: this fixture is unqualified.
    // The exact asserted refusal precedes inventory/admission, not a successful
    // checked-trace claim made with a requirement-as-grant.
    let manifest=PublicCiphertextConstructionManifest{owners:20,persistent_owners:1,input_ranks:&[1],operations:&[],capabilities:&[],
        outputs:1,retained_inputs:1,release_lengths:&[],backings:&[],other_owners:&[],phases:1,manifest_original_bytes:None};
    let before=f.storage.graph.clone();let trace=f.storage.trace.clone();
    let mut metadata=MetadataLedger::new(65536);let mut control=ControlLedger::new(&mut work);
    for _ in 0..2 {
        let remaining=control.core_work().remaining();
        let result=lower(Source::Row{plan:&plan,public:&public,original:&owners[0]},&b,&mut f.storage,&manifest,&mut control,&mut metadata,None);
        assert!(matches!(result,Err(Refusal::Work)));
        assert_eq!(control.core_work().remaining(),remaining-64);
        assert_eq!(metadata.remaining(),65536);
        assert_eq!(f.storage.graph,before);assert_eq!(f.storage.trace,trace);assert!(!f.storage.poisoned);
    }
}

#[test]
fn unwind_after_private_fill_leaves_no_checked_result_and_poisoned_storage() {
    let budget=Budget::new(1<<20).unwrap();let guard=budget.charge_external(64).unwrap();
    let raw=vec![0u8;64];let owner=PublicCiphertextOwner{identity:"encoded".into(),bytes:Some(64)};
    let encoded_state=state();
    let owners=vec![OriginalOwner{owner:&owner,backing:OriginalBacking::Bytes(&raw),original_bytes:Some(64),guard:&guard,manifest_other:0,parameter_state:Some(&encoded_state)}];
    let mut work=WorkLedger::new(65536,65536);let plan=InstructionPlan::new(Geometry::new(1,2,1).unwrap(),&mut work).unwrap();
    let public=vec![0.;8];let source=Source::Row{plan:&plan,public:&public,original:&owners[0]};
    let mut f=synthetic(&plan,&public,&owners[0]);
    let b=Bindings{rows:&f.rows,capabilities:&f.caps,values:&f.values,owners:&owners,zeros:None,budget:&budget,overlap:None,
        metadata:MetadataCharge{guard:&guard,original_bytes:None,manifest_other:0}};
    let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||{
        let _unqualified=inspect_unqualified(&source,&b,&mut f.storage).unwrap();
        panic!("unit-test unwind after private source fill; never Checked");
    }));
    assert!(result.is_err());assert!(f.storage.poisoned);
    assert!(matches!(preflight(&source,&b,&f.storage,18),Err(Refusal::Poisoned)));
}

#[test]
fn unqualified_pc_actual_ssa_and_independent_literal_release_table() {
    use pllm_core::slot_matrix_layout::{Layout,Diagonal,Limits,block_pc_physical_plan_scaled_support};
    // This fixture intentionally has NO qualified encoded operands/keys and NO
    // original fresh-output backing custody. It cannot pass lower()/admit_trace.
    // A shared dummy metadata owner is NOT a claim of permitted operand packing.
    const RELEASE:[&[usize];88]=[
        &[],&[],&[4,5],&[6],&[7],&[8],&[9],
        &[],&[],&[11,12],&[13],&[],&[15],&[14],&[17],&[16],&[18],&[19,20],&[10,21],
        &[],&[],&[23,24],&[25],&[],&[27],&[26],&[29],&[28],&[30],&[31,32],&[22,33],
        &[],&[],&[35,36],&[37],&[],&[39],&[38],&[41],&[40],&[42],&[43,44],&[34,45],
        &[],&[],&[47,48],&[49],&[50],&[51],&[52],
        &[],&[],&[54,55],&[56],&[],&[58],&[57],&[60],&[59],&[61],&[62,63],&[53,64],
        &[],&[],&[66,67],&[68],&[],&[70],&[69],&[72],&[71],&[73],&[74,75],&[65,76],
        &[],&[],&[78,79],&[80],&[],&[82],&[81],&[84],&[83],&[85],&[86,87],&[77,88],
        &[46],&[89],
    ];
    let limits=Limits{max_buffer_elements:2_000_000,max_work:100_000_000};
    let input=Layout::new(2,2,2,2,2,1,Diagonal::Lower,limits).unwrap();
    let output=Layout::new(2,2,2,2,4,1,Diagonal::Lower,limits).unwrap();
    let plan=block_pc_physical_plan_scaled_support(input,output,4096,limits).unwrap();
    let budget=Budget::new(1<<20).unwrap();
    let raw=(0..7).map(|_|vec![0u8;64]).collect::<Vec<_>>();
    let guards=(0..7).map(|_|budget.charge_external(64).unwrap()).collect::<Vec<_>>();
    let declared=["encoded","prime_one","prime_two","ct02","ct03","encoded_bias","encoded_mask"].map(|s|PublicCiphertextOwner{identity:s.into(),bytes:Some(64)});
    let mut encoded_state=state();encoded_state.shape=vec![4096];encoded_state.level=3;
    let mut mask_state=encoded_state.clone();mask_state.level=2;
    let mut bias_state=encoded_state.clone();bias_state.level=1;bias_state.scale="s4".into();
    let owners=(0..7).map(|i|OriginalOwner{owner:&declared[i],backing:OriginalBacking::Bytes(&raw[i]),
        original_bytes:Some(64),guard:&guards[i],manifest_other:i,parameter_state:match i{0=>Some(&encoded_state),3|4|5=>Some(&bias_state),6=>Some(&mask_state),_=>None}}).collect::<Vec<_>>();
    let weights=vec![f64::NAN;32];let bias=vec![f64::NAN;8];
    let source=Source::Pc{plan:&plan,weights:&weights,bias:&bias,weights_owner:&owners[0],bias_owner:&owners[0]};
    let toy=|scale:&str,level:u32|{let mut s=state();s.scale=scale.into();s.level=level;s.shape=vec![4096];s};
    let mut values=(0..4).map(|i|PublicCiphertextValue{identity:format!("v{i:02}"),owner:format!("ct{i:02}"),
        state:if i<2 {toy("s0",3)}else{toy("s4",1)}}).collect::<Vec<_>>();
    let instructions=pc_expected();
    let mut caps=Vec::new();let mut selections:Vec<(Selector,usize)>=Vec::new();let mut operations=Vec::new();let mut trace=Vec::new();
    for (i,ins) in instructions.iter().enumerate() {
        let output=match ins.selector {
            Selector::Weight{..}=>toy("s1",3),Selector::Drop{boundary:1,..}=>toy("s2",2),
            Selector::Mask{..}=>toy("s3",2),_=>values[ins.parents[0]].state.clone(),
        };
        let output=if matches!(ins.selector,Selector::Drop{boundary:2,..}){toy("s4",1)}else{output};
        values.push(PublicCiphertextValue{identity:format!("v{:02}",i+4),owner:format!("ct{:02}",i+4),state:output.clone()});
        let input_states=ins.parents[..ins.arity].iter().map(|&p|values[p].state.clone()).collect::<Vec<_>>();
        let selected=selections.iter().find_map(|&(sel,index)|{
            let c:&PublicCiphertextCapability=&caps[index];
            (sel==ins.selector&&c.inputs==input_states&&c.output==output).then_some(index)
        });
        let index=selected.unwrap_or_else(||{
            let index=caps.len();let mut c=cap(index,ins.arity);c.inputs=input_states;c.output=output.clone();
            let from=c.inputs[0].level;let to=c.output.level;
            c.execution=PublicCiphertextExecution::Ckks{compute:PublicLevelCompute{input_level:from,cost:PublicCost{identity:"compute".into(),units:Some(1)}},depth:0,
                transition:PublicLevelTransition{kind:if from==to{PublicLevelTransitionKind::Preserve}else{PublicLevelTransitionKind::Drop},from_level:from,to_level:to,
                    cost:PublicCost{identity:"transition".into(),units:Some(0)}}};
            caps.push(c);selections.push((ins.selector,index));index
        });
        let _=index;
        operations.push(PublicCiphertextGraphOperation{identity:format!("o{i:02}"),kernel_identity:"xxx".into(),primitive:PublicPrimitive::Arithmetic,
            inputs:vec!["xxx".into();ins.arity],output:PublicCiphertextValue{identity:"xxx".into(),owner:"xxxx".into(),state:output}});
        trace.push(PublicCiphertextTraceStep{operation:"xxx".into(),capability:"xxx".into(),release:vec!["xxx".into();RELEASE[i].len()]});
    }
    let empty=Vec::new();let first=vec![&owners[1]];let second=vec![&owners[1],&owners[2]];
    let make_rows=||selections.iter().map(|&(selector,capability)|{
        let c=&caps[capability];let evidence=match selector {
            Selector::Arithmetic(_)=>Evidence::Arithmetic,
            Selector::Weight{q,a,j}=>Evidence::Operand{source:Operand::Weight{source:&weights,q,a,j},encoded:&owners[0]},
            Selector::Mask{q,a,branch}=>Evidence::Operand{source:Operand::Mask{plan:&plan,q,a,branch},encoded:&owners[6]},
            Selector::Bias(q)=>Evidence::Operand{source:Operand::Bias{source:&bias,q},encoded:&owners[5]},
            Selector::Rotation{shift,..}=>Evidence::Rotation{signed_offset:shift,key:&owners[0],key_domain:"unqualified",layout:"two_slots"},
            Selector::Drop{boundary,..}=>Evidence::Drop{context:&owners[0],before:if boundary==1{&empty}else{&first},after:if boundary==1{&first}else{&second},input:&c.inputs[0],output:&c.output},
            _=>panic!("PC only"),
        };Binding{selector,capability,evidence}
    }).collect::<Vec<_>>();
    let rows=make_rows();
    let graph_owners=values.iter().map(|v|PublicCiphertextOwner{identity:v.owner.clone(),bytes:Some(64)})
        .chain([0,1,2,5,6].map(|i|declared[i].clone())).collect();
    let mut storage=Storage::new(PublicCiphertextGraph{cost_model_identity:"test".into(),cost_unit:"unit".into(),maximum_level:3,
        owners:graph_owners,persistent_owners:[0,1,2,5,6].map(|i|declared[i].identity.clone()).to_vec(),inputs:values[..4].to_vec(),operations,
        capabilities:caps.clone(),outputs:vec!["v90".into(),"v91".into()],retained_inputs:(0..4).map(|i|format!("v{i:02}")).collect()},trace);
    let reversed=vec![&owners[2],&owners[1]];
    for case in 0..9 {
        let mut negative_rows=make_rows();
        match case {
            0|1=>{
                let row=negative_rows.iter_mut().find(|r|r.selector==Selector::Mask{q:0,a:1,branch:Branch::Remainder}).unwrap();
                row.evidence=Evidence::Operand{source:Operand::Mask{plan:&plan,q:if case==0{0}else{1},a:1,
                     branch:if case==0{Branch::Wrap}else{Branch::Remainder}},encoded:&owners[6]};
            },
            2=>{
                let row=negative_rows.iter_mut().find(|r|matches!(r.selector,Selector::Drop{boundary:2,..})).unwrap();
                let c=&caps[row.capability];
                row.evidence=Evidence::Drop{context:&owners[0],before:&first,after:&reversed,input:&c.inputs[0],output:&c.output};
            },
            3=>{
                let position=negative_rows.iter().position(|r|r.selector==Selector::Bias(0)).unwrap();negative_rows.remove(position);
            },
            4=>{},
            5=>{
                let row=negative_rows.iter_mut().find(|r|matches!(r.selector,Selector::Drop{boundary:1,..})).unwrap();row.evidence=Evidence::Arithmetic;
            },
            6=>{
                let row=negative_rows.iter_mut().find(|r|r.selector==Selector::Bias(0)).unwrap();
                // Mask encoding is not beta-level, beta-scale bias authority.
                row.evidence=Evidence::Operand{source:Operand::Bias{source:&bias,q:0},encoded:&owners[6]};
            },
            7=>{
                 let row=negative_rows.iter_mut().find(|r|r.selector==Selector::Weight{q:0,a:0,j:0}).unwrap();
                 row.evidence=Evidence::Operand{source:Operand::Weight{source:&weights,q:0,a:0,j:0},encoded:&owners[5]};
             },
            _=>{
                let row=negative_rows.iter_mut().find(|r|matches!(r.selector,Selector::Rotation{shift:3,..})).unwrap();
                if let Evidence::Rotation{signed_offset,..}=&mut row.evidence {*signed_offset=-3;}
            },
        }
        let b=Bindings{rows:&negative_rows,capabilities:&caps,values:&values,owners:&owners,
            zeros:if case==4{None}else{Some([&owners[3],&owners[4]])},budget:&budget,overlap:None,
            metadata:MetadataCharge{guard:&guards[0],original_bytes:None,manifest_other:0}};
        let before=storage.graph.clone();let trace=storage.trace.clone();
        assert!(preflight(&source,&b,&storage,88).is_err(),"negative case {case}");
        assert_eq!(storage.graph,before);assert_eq!(storage.trace,trace);assert!(!storage.poisoned);
    }
    // Retained roots may share one ORIGINAL full backing at identical state;
    // Full zero declarations are checked before row selection or private writes.
    for case in 0..12 {
        let mut wrong=bias_state.clone();
        match case {
            1=>wrong.level=2,2=>wrong.scale="foreign".into(),3=>wrong.key_domain="foreign".into(),
            4=>wrong.layout="foreign".into(),5=>wrong.shape=vec![2048],
            6=>wrong.error.model_identity="foreign".into(),7=>wrong.error.bound=Some(1),
            8=>wrong.scheme="foreign".into(),_=>{},
        }
        let negative_owners=(0..7).map(|i|OriginalOwner{owner:&declared[i],backing:OriginalBacking::Bytes(&raw[i]),
            original_bytes:Some(64),guard:&guards[i],manifest_other:i,
            parameter_state:if i==3 {if case==0{None}else{Some(&wrong)}}else{owners[i].parameter_state}}).collect::<Vec<_>>();
        let zeros=match case {
            9=>[&negative_owners[4],&negative_owners[3]],
            10=>[&negative_owners[0],&negative_owners[4]],
            11=>[&owners[3],&negative_owners[4]], // equal label/state, wrong original member
            _=>[&negative_owners[3],&negative_owners[4]],
        };
        let b=Bindings{rows:&rows,capabilities:&caps,values:&values,owners:&negative_owners,zeros:Some(zeros),budget:&budget,overlap:None,
            metadata:MetadataCharge{guard:&guards[0],original_bytes:None,manifest_other:0}};
        let before=storage.graph.clone();let trace=storage.trace.clone();
        assert!(matches!(preflight(&source,&b,&storage,88),Err(e) if e==if case==0||case>=9 {
            Refusal::MissingOperandBinding}else{Refusal::State}),"zero case {case}");
        assert_eq!(storage.graph,before);assert_eq!(storage.trace,trace);assert!(!storage.poisoned);
    }
    // no computed value may use that backing, including an apparent suffix.
    let mut aliased_values=values.clone();aliased_values[1].owner=aliased_values[0].owner.clone();
    let mut aliased_storage=Storage::new(storage.graph.clone(),storage.trace.clone());
    aliased_storage.graph.inputs[1].owner=aliased_values[1].owner.clone();
    aliased_storage.graph.owners.retain(|o|o.identity!="ct01");
    {
        let b=Bindings{rows:&rows,capabilities:&caps,values:&aliased_values,owners:&owners,zeros:Some([&owners[3],&owners[4]]),budget:&budget,overlap:None,
            metadata:MetadataCharge{guard:&guards[0],original_bytes:None,manifest_other:0}};
        assert!(preflight(&source,&b,&aliased_storage,88).is_ok());
    }
    aliased_values[4].owner=aliased_values[0].owner.clone();
    {
        let b=Bindings{rows:&rows,capabilities:&caps,values:&aliased_values,owners:&owners,zeros:Some([&owners[3],&owners[4]]),budget:&budget,overlap:None,
            metadata:MetadataCharge{guard:&guards[0],original_bytes:None,manifest_other:0}};
        assert!(preflight(&source,&b,&aliased_storage,88).is_err());assert!(!aliased_storage.poisoned);
    }
    let b=Bindings{rows:&rows,capabilities:&caps,values:&values,owners:&owners,zeros:Some([&owners[3],&owners[4]]),budget:&budget,overlap:None,
        metadata:MetadataCharge{guard:&guards[0],original_bytes:None,manifest_other:0}};
    let unqualified=inspect_unqualified(&source,&b,&mut storage).unwrap();
    for (i,op) in unqualified.storage.graph.operations.iter().enumerate(){
         let ins=instructions[i];assert_eq!(op.output,values[i+4]);assert_eq!(op.primitive,PublicPrimitive::Arithmetic);
         assert_eq!(op.inputs,ins.parents[..ins.arity].iter().map(|&p|values[p].identity.clone()).collect::<Vec<_>>());
         assert_eq!(unqualified.storage.trace[i].release,RELEASE[i].iter().map(|&p|values[p].identity.clone()).collect::<Vec<_>>());
         let input_states=ins.parents[..ins.arity].iter().map(|&p|values[p].state.clone()).collect::<Vec<_>>();
         let index=selections.iter().find_map(|&(sel,index)|
             (sel==ins.selector&&caps[index].inputs==input_states&&caps[index].output==values[i+4].state).then_some(index)).unwrap();
         assert_eq!(op.kernel_identity,caps[index].kernel_identity);
         assert_eq!(unqualified.storage.trace[i].operation,op.identity);
         assert_eq!(unqualified.storage.trace[i].capability,caps[index].identity);
         assert_eq!(caps[index].inputs,input_states);assert_eq!(caps[index].output,op.output.state);
     }
    assert!(selections.iter().enumerate().any(|(i,&(s,k))|selections[..i].iter().any(|&(t,l)|
        s==t&&k!=l&&caps[k].inputs!=caps[l].inputs)));
    for (i,&(s,k)) in selections.iter().enumerate() { for &(t,l) in &selections[..i] {
        if s!=t {assert_ne!(k,l);assert_ne!(caps[k].kernel_identity,caps[l].kernel_identity);}
    }}
    assert_eq!(unqualified.storage.trace.iter().map(|s|s.release.len()).sum::<usize>(),86);
    assert_eq!(unqualified.storage.graph.outputs,["v90","v91"]);
    assert_eq!(unqualified.storage.graph.inputs.len(),4);
}
