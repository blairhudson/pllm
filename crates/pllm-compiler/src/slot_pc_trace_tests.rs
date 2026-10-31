//! Independent literal arithmetic expansion. SOURCE only; never HE admission.
use super::*;
use pllm_core::slot_matrix_layout::{block_pc_physical_plan, block_pc_physical_plan_scaled_support,
    Diagonal, Layout, Limits, PhysicalMaskBranch};
const N:usize=usize::MAX;
// Four roots 0,1,2,3. Outputs 4..91. Hand derived, not stages()/materializers.
const OPS:[(char,usize,usize);88]=[
    ('W',0,N),('W',1,N),('+',4,5),('D',6,N),('M',7,N),('D',8,N),('+',2,9),
    ('W',0,N),('W',1,N),('+',11,12),('D',13,N),('M',14,N),('D',15,N),('M',14,N),('D',17,N),('R',16,N),('R',18,N),('+',19,20),('+',10,21),
    ('W',0,N),('W',1,N),('+',23,24),('D',25,N),('M',26,N),('D',27,N),('M',26,N),('D',29,N),('R',28,N),('R',30,N),('+',31,32),('+',22,33),
    ('W',0,N),('W',1,N),('+',35,36),('D',37,N),('M',38,N),('D',39,N),('M',38,N),('D',41,N),('R',40,N),('R',42,N),('+',43,44),('+',34,45),
    ('W',0,N),('W',1,N),('+',47,48),('D',49,N),('M',50,N),('D',51,N),('+',3,52),
    ('W',0,N),('W',1,N),('+',54,55),('D',56,N),('M',57,N),('D',58,N),('M',57,N),('D',60,N),('R',59,N),('R',61,N),('+',62,63),('+',53,64),
    ('W',0,N),('W',1,N),('+',66,67),('D',68,N),('M',69,N),('D',70,N),('M',69,N),('D',72,N),('R',71,N),('R',73,N),('+',74,75),('+',65,76),
    ('W',0,N),('W',1,N),('+',78,79),('D',80,N),('M',81,N),('D',82,N),('M',81,N),('D',84,N),('R',83,N),('R',85,N),('+',86,87),('+',77,88),
    ('B',46,N),('B',89,N),
];
fn plan(i:usize,o:usize,scaled:bool)->BlockPcPhysicalPlan {
    let limits=Limits{max_buffer_elements:2_000_000,max_work:100_000_000};
    let input=Layout::new(2,2,2,2,i/2,1,Diagonal::Lower,limits).unwrap();
    let out=Layout::new(2,2,2,2,o/2,1,Diagonal::Lower,limits).unwrap();
    if scaled {block_pc_physical_plan_scaled_support(input,out,4096,limits)}
    else {block_pc_physical_plan(input,out,4096,limits)}.unwrap()
}
#[test]
fn literal_88_parents_four_roots_two_finals_and_census_both_directions() {
    for (iw,ow) in [(4,8),(8,4)] {
        let source=plan(iw,ow,true); let mut cursor=Cursor::new(&source);
        let mut census=[0;9];
        for (i,&(tag,left,right)) in OPS.iter().enumerate() {
            let ins=cursor.next().unwrap().unwrap();
            assert_eq!(ins.parents,[left,right],"ordinal {}",i+4);
            assert_eq!(ins.arity,if right==N {1}else{2});
            let actual=match ins.selector {
                Selector::Weight{..}=>{census[0]+=1;'W'},
                Selector::Drop{boundary:1,..}=>{census[2]+=1;'D'},
                Selector::Drop{boundary:2,..}=>{census[4]+=1;'D'},
                Selector::Mask{..}=>{census[3]+=1;'M'},
                Selector::Rotation{..}=>{census[5]+=1;'R'},
                Selector::Arithmetic(_)=>{
                    if matches!(i+4,6|13|25|37|49|56|68|80){census[1]+=1;}
                    else if matches!(i+4,21|33|45|64|76|88){census[6]+=1;}
                    else {census[7]+=1;} '+'},
                Selector::Bias(q)=>{assert_eq!(q,i+4-90);census[8]+=1;'B'},
                _=>panic!("foreign selector"),
            };
            assert_eq!(actual,tag);
        }
        assert_eq!(census,[16,8,8,14,14,12,6,8,2]);
        assert_eq!(cursor.next(),Ok(None)); assert_eq!(cursor.next(),Ok(None));
    }
}
#[test]
fn exact_signed_keys_and_mask_coordinates_not_friendly_labels() {
    let source=plan(4,8,true); let mut cursor=Cursor::new(&source);
    let mut rotations=Vec::new(); let mut masks=Vec::new(); let mut weights=Vec::new();
    while let Some(ins)=cursor.next().unwrap() {
        match ins.selector {
            Selector::Rotation{q,a,shift}=>rotations.push((q,a,shift)),
            Selector::Mask{q,a,branch}=>masks.push((q,a,branch)),
            Selector::Weight{q,a,j}=>weights.push((q,a,j)),_=>{},
        }
    }
    assert_eq!(rotations,[(0,1,3),(0,1,-1),(0,2,2),(0,2,-2),(0,3,1),(0,3,-3),
        (1,1,3),(1,1,-1),(1,2,2),(1,2,-2),(1,3,1),(1,3,-3)]);
    use Branch::*;
    assert_eq!(masks,[(0,0,Support),(0,1,Wrap),(0,1,Remainder),(0,2,Wrap),(0,2,Remainder),(0,3,Wrap),(0,3,Remainder),
        (1,0,Support),(1,1,Wrap),(1,1,Remainder),(1,2,Wrap),(1,2,Remainder),(1,3,Wrap),(1,3,Remainder)]);
    assert_eq!(weights,[(0,0,0),(0,0,1),(0,1,0),(0,1,1),(0,2,0),(0,2,1),(0,3,0),(0,3,1),
        (1,0,0),(1,0,1),(1,1,0),(1,1,1),(1,2,0),(1,2,1),(1,3,0),(1,3,1)]);
}
#[test]
fn independent_coordinate_and_support_lane_tables() {
    for (iw,ow) in [(4,8),(8,4)] {
        let source=plan(iw,ow,true);
        assert_eq!(source.weight_coordinate(0,1,0,0).unwrap(),Some((0,2)));
        assert_eq!(source.weight_coordinate(0,1,0,1).unwrap(),if iw==4 {Some((2,4))}else{None});
        assert_eq!(source.weight_coordinate(0,1,0,3).unwrap(),if iw==8 {Some((6,0))}else{None});
        assert_eq!(source.output_coordinate(0,0).unwrap(),Some((0,0)));
        assert_eq!(source.output_coordinate(1,0).unwrap(),Some((0,1)));
        for (a,wrap) in [(1,&[3usize,7][..]),(2,&[2,3,6,7][..]),(3,&[1,2,3,5,6,7][..])] {
            for lane in 0..4096 {
                assert_eq!(source.branch_mask_value(a,PhysicalMaskBranch::Wrap,lane).unwrap(),wrap.contains(&lane));
                assert_eq!(source.branch_mask_value(a,PhysicalMaskBranch::Remainder,lane).unwrap(),lane<8&&!wrap.contains(&lane));
            }
        }
    }
}
#[test]
fn legacy_refuses_without_emitting_a_descriptor() {
    let source=plan(4,8,false); let mut cursor=Cursor::new(&source);
    assert_eq!(cursor.next(),Err(Refusal::LegacyUnresolved)); assert_eq!(cursor.output,4);
}
