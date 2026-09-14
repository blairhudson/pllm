//! Native microbenchmark runner. All timed arithmetic is Rust; fixtures are public.
use pllm_arithmetic::{matmul_scalar,matmul_tiled,Ring};
use std::{hint::black_box,time::Instant};

pub fn benchmark(dim:usize,lanes:usize,repeats:usize,bits:u8,tile:usize)->Result<String,String>{
    if !(1..=4096).contains(&dim)||!(1..=128).contains(&lanes)||!(3..=1000).contains(&repeats)||tile==0 {return Err("bounds: dim 1..4096, lanes 1..128, repetitions 3..1000, tile > 0".into())}
    let ring=Ring::new(bits).map_err(|e|e.to_string())?;
    // Reproducible public synthetic fixtures. NEVER use this as a mask/key generator.
    let w:Vec<i8>=(0..dim*dim).map(|i|(i%15) as i8-7).collect();
    let x:Vec<u32>=(0..dim*lanes).map(|i|ring.reduce((i as u64).wrapping_mul(2654435761))).collect();
    let reference=matmul_scalar(ring,&w,&x,dim,dim,lanes).map_err(|e|e.to_string())?;
    let candidate=matmul_tiled(ring,&w,&x,dim,dim,lanes,tile).map_err(|e|e.to_string())?;
    if reference!=candidate{return Err("exact-kernel mismatch".into())}
    for _ in 0..2 {black_box(matmul_scalar(ring,&w,&x,dim,dim,lanes).map_err(|e|e.to_string())?);black_box(matmul_tiled(ring,&w,&x,dim,dim,lanes,tile).map_err(|e|e.to_string())?);}
    let mut scalar=Vec::new();let mut tiled=Vec::new();
    for rep in 0..repeats {
        for offset in 0..2 {let which=(rep+offset)%2;let start=Instant::now();
            let value=if which==0 {matmul_scalar(ring,&w,&x,dim,dim,lanes)}else{matmul_tiled(ring,&w,&x,dim,dim,lanes,tile)}.map_err(|e|e.to_string())?;
            black_box(value);let ns=start.elapsed().as_nanos();if which==0{scalar.push(ns)}else{tiled.push(ns)}
        }
    }
    fn arr(v:&[u128])->String{v.iter().map(|n|n.to_string()).collect::<Vec<_>>().join(",")}
    Ok(format!("{{\"schema_version\":\"pllm.kernel_run.v1\",\"scope\":\"public_synthetic_modular_matrix\",\"origin\":\"native_executed\",\"dim\":{dim},\"lanes\":{lanes},\"bits\":{bits},\"tile\":{tile},\"repetitions\":{repeats},\"exact_coordinates\":{},\"scalar_ns\":[{}],\"tiled_ns\":[{}],\"allocation_included\":true,\"whole_model_tps\":null,\"cryptographic_security_claim\":null}}",reference.len(),arr(&scalar),arr(&tiled)))
}
#[cfg(test)]mod tests{#[test]fn rejects_bad_dimensions(){assert!(super::benchmark(0,1,3,16,4).is_err());}#[test]fn smoke(){assert!(super::benchmark(8,2,3,16,4).unwrap().contains("exact_coordinates"));}}
