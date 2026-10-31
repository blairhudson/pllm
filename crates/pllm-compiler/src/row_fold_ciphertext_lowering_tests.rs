//! Independent SOURCE tables. All ciphertext-looking metadata is UNQUALIFIED.
use super::*;
use pllm_core::row_fold::{Geometry, WorkLedger};

// Literal hand-derived block=2,width=1 table, NOT generated with program().
const TABLE: [(Selector, usize, Option<usize>); 18] = [
    (Selector::RowVector(0),0,None), (Selector::RowRotation(1),1,None),
    (Selector::RowVector(2),2,None), (Selector::RowRotation(-1),1,None),
    (Selector::RowVector(3),4,None), (Selector::Arithmetic(Arithmetic::Add),3,Some(5)),
    (Selector::Arithmetic(Arithmetic::Add),1,Some(6)), (Selector::RowVector(1),1,None),
    (Selector::Arithmetic(Arithmetic::Subtract),8,Some(7)), (Selector::RowVector(0),9,None),
    (Selector::Arithmetic(Arithmetic::Square),10,None), (Selector::RowRotation(1),11,None),
    (Selector::RowVector(2),12,None), (Selector::RowRotation(-1),11,None),
    (Selector::RowVector(3),14,None), (Selector::Arithmetic(Arithmetic::Add),13,Some(15)),
    (Selector::Arithmetic(Arithmetic::Add),11,Some(16)), (Selector::RowVector(0),17,None),
];

#[test]
fn literal_eighteen_operations_preserve_order_and_unary_square() {
    let mut work = WorkLedger::new(65536,65536);
    let plan = InstructionPlan::new(Geometry::new(3,2,1).unwrap(), &mut work).unwrap();
    for (i, &(selector,left,right)) in TABLE.iter().enumerate() {
        assert_eq!(instruction(&plan,i).unwrap(),Some(Instruction {
            selector, parents: [left,right.unwrap_or(usize::MAX)], arity: if right.is_some(){2}else{1},
        }));
    }
    assert_eq!(instruction(&plan,18),Ok(None));
}

#[test]
fn full_original_source_census_126_not_rmsnorm() {
    let mut work = WorkLedger::new(65536,65536);
    let before = work.remaining();
    let plan = InstructionPlan::new(Geometry::new(32,1024,768).unwrap(), &mut work).unwrap();
    assert_eq!(before-work.remaining(),8*126+32);
    let mut counts = [0;5]; let mut rotations = std::collections::BTreeSet::new();
    for i in 0..126 {
        let op=instruction(&plan,i).unwrap().unwrap();
        match op.selector {
            Selector::RowRotation(d)=>{counts[0]+=1; rotations.insert(d);},
            Selector::RowVector(_)=>counts[1]+=1,
            Selector::Arithmetic(Arithmetic::Add)=>counts[2]+=1,
            Selector::Arithmetic(Arithmetic::Subtract)=>counts[3]+=1,
            Selector::Arithmetic(Arithmetic::Square)=>{counts[4]+=1;assert_eq!(op.arity,1);},
            _=>panic!("foreign source selector"),
        }
    }
    assert_eq!(counts,[40,44,40,1,1]); assert_eq!(rotations.len(),20);
    assert_eq!(instruction(&plan,126),Ok(None));
}

#[test]
fn constructor_exact_short_and_same_ledger_retry_no_clear_arrays() {
    let required=8*126+32;
    let mut short=WorkLedger::new(required-1,required);
    assert!(InstructionPlan::new(Geometry::new(1,2,1).unwrap(),&mut short).is_err());
    assert_eq!(short.remaining(),required-1);
    let mut exact=WorkLedger::new(required,required);
    let _plan=InstructionPlan::new(Geometry::new(1,2,1).unwrap(),&mut exact).unwrap();
    assert_eq!(exact.remaining(),0);
    assert!(InstructionPlan::new(Geometry::new(1,2,1).unwrap(),&mut exact).is_err());
    assert_eq!(exact.remaining(),0);
}

#[test]
fn independent_integer_energy_and_cross_row_padding_sentinels() {
    // Evaluate ONLY the independent literal table. Expected algebra is separately
    // dense integer centered energy: sum_i (w*x_i-sum_j x_j)^2. No normalization.
    let rows=[[2i64,999],[-3,-777],[0,555]];
    let mut values=[[0i64;6];19];
    values[0]=[2,999,-3,-777,0,555];
    for (i,&(selector,left,right)) in TABLE.iter().enumerate() {
        for lane in 0..6 {
            let x=values[left][lane];
            values[i+1][lane]=match selector {
                Selector::RowRotation(d)=>values[left][(lane as isize+d).rem_euclid(6) as usize],
                Selector::RowVector(v)=>x*match v {0=>i64::from(lane%2==0),1=>1,2=>i64::from(lane%2==0),3=>i64::from(lane%2==1),_=>panic!()},
                Selector::Arithmetic(Arithmetic::Add)=>x+values[right.unwrap()][lane],
                Selector::Arithmetic(Arithmetic::Subtract)=>x-values[right.unwrap()][lane],
                Selector::Arithmetic(Arithmetic::Square)=>x*x,_=>panic!(),
            };
        }
    }
    for (r,row) in rows.iter().enumerate() {
        let dense=(row[0]-row[0]).pow(2);
        assert_eq!(values[18][2*r],dense); assert_eq!(values[18][2*r+1],0);
    }
}

#[test]
fn nonzero_dense_integer_energy_excludes_padding_and_cross_row_wrap() {
    let mut work=WorkLedger::new(65536,65536);
    let plan=InstructionPlan::new(Geometry::new(2,4,3).unwrap(),&mut work).unwrap();
    let mut values=[[0i64;8];31];
    values[0]=[2,-1,3,100_003,4,0,-2,-70_001];
    for i in 0..30 {
        let op=instruction(&plan,i).unwrap().unwrap();
        for lane in 0..8 {
            let x=values[op.parents[0]][lane];
            values[i+1][lane]=match op.selector {
                Selector::RowRotation(d)=>values[op.parents[0]][(lane as isize+d).rem_euclid(8) as usize],
                Selector::RowVector(v)=>x*match v {
                    0=>i64::from(lane%4<3),1=>3,
                    2=>i64::from(lane%4<3),3=>i64::from(lane%4>=3),
                    4=>i64::from(lane%4<2),5=>i64::from(lane%4>=2),_=>panic!(),
                },
                Selector::Arithmetic(Arithmetic::Add)=>x+values[op.parents[1]][lane],
                Selector::Arithmetic(Arithmetic::Subtract)=>x-values[op.parents[1]][lane],
                Selector::Arithmetic(Arithmetic::Square)=>x*x,_=>panic!(),
            };
        }
    }
    assert_eq!(values[30],[78,78,78,0,168,168,168,0]);
}
