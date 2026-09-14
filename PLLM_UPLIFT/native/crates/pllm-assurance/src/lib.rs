//! Finite ideal-model checks and attacks on deliberately weakened constructions.
//! These are neither computational-security proofs nor audits of a production runtime.
use pllm_inventory::Record;
#[derive(Debug)]pub struct Finding {pub id:&'static str,pub status:&'static str,pub scope:&'static str}
pub fn run()->Result<Vec<Finding>,String> {
    let q=8usize;let w=3usize;let mut reference=None;
    for x in 0..q {
        let mut hist=vec![0u32;q*q];
        for r in 0..q {for s in 0..q {let u=(x+q-r)%q;let c=(w*r+q-s)%q;hist[u*q+c]+=1;}}
        if hist.iter().any(|&n|n!=1){return Err("nonuniform ideal transcript".into())}
        if let Some(ref old)=reference {if &hist!=old{return Err("ideal view differs by input".into())}}else{reference=Some(hist)}
    }
    let (x0,x1,r,q)=(13i64,27i64,71i64,256i64);
    let d0=(x0-r).rem_euclid(q);let d1=(x1-r).rem_euclid(q);
    if (d1-d0).rem_euclid(q)!=(x1-x0).rem_euclid(q){return Err("reuse negative control failed".into())}
    let mut item=Record::new(1);item.bind(b"first").map_err(|e|e.to_string())?;item.expose().map_err(|e|e.to_string())?;
    if item.bind(b"different").is_ok(){return Err("record permitted changed semantic input".into())}
    Ok(vec![
        Finding{id:"ideal_joint_view_q8",status:"finite_model_checked",scope:"uniform independent masks; one scalar; q=8; not PRG security"},
        Finding{id:"mask_reuse",status:"counterexample_expected",scope:"deliberately reused additive mask leaks input differences"},
        Finding{id:"assignment",status:"unit_property_checked",scope:"in-memory allocation; excludes crashes, clones and rollback"}
    ])
}
pub fn json()->Result<String,String>{let fs=run()?;Ok(format!("{{\"schema_version\":\"pllm.assurance_smoke.v1\",\"findings\":[{}],\"whole_protocol_privacy_proven\":false}}",fs.iter().map(|f|format!("{{\"id\":\"{}\",\"status\":\"{}\",\"scope\":\"{}\"}}",f.id,f.status,f.scope)).collect::<Vec<_>>().join(",")))}
#[cfg(test)]mod tests{#[test]fn checks_execute(){assert_eq!(super::run().unwrap().len(),3);}}
