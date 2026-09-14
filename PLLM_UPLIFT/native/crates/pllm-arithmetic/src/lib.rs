//! Exact reference/native kernels. No secret generation or cryptographic protocol here.
use std::fmt;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub &'static str);
impl fmt::Display for Error { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.0) } }
impl std::error::Error for Error {}
#[derive(Clone, Copy, Debug)]
pub struct Ring { bits: u8, mask: u64 }
impl Ring {
    pub fn new(bits: u8) -> Result<Self, Error> {
        if !(1..=32).contains(&bits) { return Err(Error("supported ring widths are 1..=32")); }
        Ok(Self { bits, mask: (1u64 << bits)-1 })
    }
    pub fn bits(self) -> u8 { self.bits }
    pub fn reduce(self, value: u64) -> u32 { (value & self.mask) as u32 }
    pub fn add(self, a: u32, b: u32) -> u32 { self.reduce((a as u64)+(b as u64)) }
    pub fn sub(self, a: u32, b: u32) -> u32 { self.reduce((a as u64).wrapping_sub(b as u64)) }
    pub fn mul(self, a: u32, b: u32) -> u32 { self.reduce((a as u64)*(b as u64)) }
    pub fn signed(self, value: u32) -> i64 {
        let v=self.reduce(value as u64) as i64;
        if v >= (1i64 << (self.bits-1)) { v-(1i64 << self.bits) } else { v }
    }
    pub fn encode_signed(self, value: i64) -> u32 { self.reduce(value as u64) }
    fn madd_weight(self, acc: u32, w: i8, x: u32) -> u32 {
        self.reduce((acc as u64).wrapping_add((w as i64 * x as i64) as u64))
    }
}
fn validate_shape(w: &[i8], x: &[u32], m: usize, n: usize, lanes: usize) -> Result<usize,Error> {
    if m==0 || n==0 || lanes==0 { return Err(Error("empty shape")); }
    if m.checked_mul(n)!=Some(w.len()) || n.checked_mul(lanes)!=Some(x.len()) { return Err(Error("shape mismatch or overflow")); }
    m.checked_mul(lanes).ok_or(Error("output shape overflow"))
}
/// All lanes are processed in native code. Caller never crosses FFI per element.
pub fn matmul_scalar(r: Ring, w: &[i8], x: &[u32], m: usize, n: usize, lanes: usize) -> Result<Vec<u32>,Error> {
    let size=validate_shape(w,x,m,n,lanes)?; let mut out=vec![0;size];
    for i in 0..m { for lane in 0..lanes { let mut a=0; for k in 0..n { a=r.madd_weight(a,w[i*n+k],x[k*lanes+lane]); } out[i*lanes+lane]=a; } }
    Ok(out)
}
/// Cache-friendly multi-lane reference; SIMD/GPU backends must match these semantics.
pub fn matmul_tiled(r: Ring, w: &[i8], x: &[u32], m: usize, n: usize, lanes: usize, tile: usize) -> Result<Vec<u32>,Error> {
    if tile==0 { return Err(Error("tile must be positive")); }
    let size=validate_shape(w,x,m,n,lanes)?; let mut out=vec![0;size];
    for start in (0..n).step_by(tile) {
        let end=start.saturating_add(tile).min(n);
        for i in 0..m { for k in start..end { let weight=w[i*n+k]; for lane in 0..lanes {
            let ix=i*lanes+lane; out[ix]=r.madd_weight(out[ix],weight,x[k*lanes+lane]);
        } } }
    }
    Ok(out)
}
pub fn pack(values: &[u32], bits: u8) -> Result<Vec<u8>,Error> {
    let r=Ring::new(bits)?;
    let total=values.len().checked_mul(bits as usize).ok_or(Error("size overflow"))?;
    let byte_len=total.checked_add(7).ok_or(Error("size overflow"))?/8;
    let mut out=vec![0u8;byte_len]; let mut pos=0usize;
    for &value in values {
        if value as u64 > r.mask { return Err(Error("value does not fit wire width")); }
        for j in 0..bits { out[pos/8] |= (((value>>j)&1) as u8) << (pos%8); pos+=1; }
    }
    Ok(out)
}
pub fn unpack(bytes: &[u8], count: usize, bits: u8) -> Result<Vec<u32>,Error> {
    Ring::new(bits)?;
    let total=count.checked_mul(bits as usize).ok_or(Error("size overflow"))?;
    if total.checked_add(7).ok_or(Error("size overflow"))?/8 != bytes.len() { return Err(Error("wire length mismatch")); }
    if total%8!=0 && bytes.last().copied().unwrap_or(0) >> (total%8) !=0 { return Err(Error("noncanonical padding")); }
    let mut out=vec![0;count]; let mut pos=0usize;
    for value in &mut out { for j in 0..bits { *value |= (((bytes[pos/8]>>(pos%8))&1) as u32)<<j; pos+=1; } }
    Ok(out)
}
/// Exact CRT for the initial small-prime experiments; not a nonlinear/base-extension gadget.
pub fn crt_signed(a: u32,b: u32,p: u16,q: u16) -> Result<i64,Error> {
    fn prime(x:u16)->bool { x>=3 && x%2!=0 && (3..=((x as f64).sqrt() as u16)).step_by(2).all(|d|x%d!=0) }
    if p==q || !prime(p) || !prime(q) || a>=p as u32 || b>=q as u32 { return Err(Error("distinct odd primes and canonical residues required")); }
    let (mut old_r,mut r)=(p as i64,q as i64); let (mut old_s,mut s)=(1i64,0i64);
    while r!=0 { let d=old_r/r; (old_r,r)=(r,old_r-d*r); (old_s,s)=(s,old_s-d*s); }
    if old_r!=1 { return Err(Error("noncoprime moduli")); }
    let t=((b as i64-a as i64)*old_s).rem_euclid(q as i64);
    let modulus=p as i64*q as i64; let value=a as i64+p as i64*t;
    Ok(if value>modulus/2 { value-modulus } else { value })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn ring_boundary() { let r=Ring::new(32).unwrap(); assert_eq!(r.add(u32::MAX,1),0); assert_eq!(r.signed(u32::MAX),-1); }
    #[test] fn ring_identity() { for bits in [8,16,24,32] { let r=Ring::new(bits).unwrap(); for x in 0..100u32 { let mask=77;let s=219;let w=13;let c=r.sub(r.mul(w,mask),s);let v=r.add(r.mul(w,r.sub(x,mask)),c);assert_eq!(r.add(v,s),r.mul(w,x)); } } }
    #[test] fn tiled_parity() { let w:Vec<i8>=(0..63).map(|i|(i%15) as i8-7).collect();let x:Vec<u32>=(0..9*18).map(|i|(i as u32).wrapping_mul(345791)).collect();for bits in [16,24,32] {let r=Ring::new(bits).unwrap();assert_eq!(matmul_scalar(r,&w,&x,7,9,18).unwrap(),matmul_tiled(r,&w,&x,7,9,18,4).unwrap());} }
    #[test] fn wire_roundtrip() { for bits in 1..=32 {let r=Ring::new(bits).unwrap();let v:Vec<u32>=(0..65).map(|i|r.reduce(i*7727)).collect();assert_eq!(unpack(&pack(&v,bits).unwrap(),v.len(),bits).unwrap(),v);} }
    #[test] fn padding_rejected() {assert!(unpack(&[255],1,1).is_err());}
    #[test] fn crt_exhaustive() { for x in -30245i64..=30245 {assert_eq!(crt_signed(x.rem_euclid(251) as u32,x.rem_euclid(241) as u32,251,241).unwrap(),x);} }
    #[test] fn invalid_shape() {assert!(matmul_scalar(Ring::new(8).unwrap(),&[],&[],1,1,1).is_err());}
}
