"""BFV preparation from OS masks through valid encrypted results and decryption."""
from dataclasses import dataclass
import time
import numpy as np
import _sealapi_cpp as seal
from .bridge import Bridge
from .coefficients import CoefficientGEMM
from .randomness import uniform_residues

@dataclass(frozen=True)
class Profile:
    degree:int=2048
    modulus:int=2097169
    bits:int=54
    def context(self):
        p=seal.EncryptionParameters(seal.SCHEME_TYPE.BFV);p.set_poly_modulus_degree(self.degree)
        p.set_coeff_modulus(seal.CoeffModulus.Create(self.degree,[self.bits]));p.set_plain_modulus(self.modulus)
        ctx=seal.SEALContext(p,True,seal.SEC_LEVEL_TYPE.TC128)
        if not ctx.parameters_set():raise ValueError(ctx.parameters_error_message())
        return ctx

class Client:
    def __init__(self,profile):
        self.profile=profile;self.context=profile.context();kg=seal.KeyGenerator(self.context)
        self.secret_key=kg.secret_key();self.encryptor=seal.Encryptor(self.context,self.secret_key)
        self.decryptor=seal.Decryptor(self.context,self.secret_key);self.bridge=Bridge(self.context)
    def encrypt(self,r):
        batch,width=r.shape
        if not 1<=batch<=self.profile.degree:raise ValueError('invalid batch')
        a=np.empty((width,2*self.profile.degree),dtype=np.uint64)
        for i in range(width):
            ct=seal.Ciphertext();self.encryptor.encrypt_symmetric(self.bridge.plaintext(r[:,i]),ct)
            a[i],template=self.bridge.ciphertext_array(ct)
        ct=seal.Ciphertext();self.encryptor.encrypt_symmetric(self.bridge.plaintext(np.zeros(1,dtype=np.int64)),ct)
        zero,_=self.bridge.ciphertext_array(ct)
        return a,zero,template
    def decrypt(self,a,template,count):
        out=np.empty((count,len(a)),dtype=np.uint32);noise=100000
        for j,row in enumerate(a):
            ct=self.bridge.array_ciphertext(row,template);pt=seal.Plaintext();self.decryptor.decrypt(ct,pt)
            out[:,j]=self.bridge.plaintext_array(pt,count)
            noise=min(noise,self.decryptor.invariant_noise_budget(ct))
        return out,noise
    def close(self):self.bridge.close()

def prepare(w,profile,batch=2048,chunk=128,verify_rows=16):
    if w.dtype!=np.int8 or w.ndim!=2:raise ValueError('int8 matrix required')
    # Bound a true signed dot product, not merely the modular reference.
    bound=7*max(int(np.abs(row.astype(np.int16)).sum()) for row in w)
    if 2*bound>=profile.modulus:raise ValueError('modulus too small for signed W4A4 result')
    t=time.perf_counter();c=Client(profile);setup=time.perf_counter()-t
    t=time.perf_counter();r=uniform_residues(profile.modulus,(batch,w.shape[1]));random_s=time.perf_counter()-t
    t=time.perf_counter();a,zero,template=c.encrypt(r);enc=time.perf_counter()-t
    t=time.perf_counter();executor=CoefficientGEMM(a,c.bridge.q,zero);digits=time.perf_counter()-t
    wr=np.empty((batch,len(w)),dtype=np.uint32);evaluation=decryption=validation=0.;noise=100000
    ids=np.unique(np.linspace(0,batch-1,min(batch,verify_rows)).astype(int))
    for start in range(0,len(w),chunk):
        part=w[start:start+chunk]
        t=time.perf_counter();out=executor.evaluate(part);evaluation+=time.perf_counter()-t
        t=time.perf_counter();plain,budget=c.decrypt(out,template,batch);decryption+=time.perf_counter()-t
        wr[:,start:start+len(part)]=plain;noise=min(noise,budget)
        t=time.perf_counter();expected=(r[ids].astype(np.int64)@part.astype(np.int64).T)%profile.modulus
        if not np.array_equal(expected,plain[ids]):raise AssertionError('decrypted output differs')
        validation+=time.perf_counter()-t
    total=random_s+enc+digits+evaluation+decryption
    record=dict(shape=[w.shape[1],len(w)],batch=batch,degree=profile.degree,plain_modulus=profile.modulus,
      coefficient_modulus=c.bridge.q,coefficient_bits=profile.bits,setup_s=setup,random_s=random_s,
      client_encrypt_export_s=enc,server_digits_s=digits,server_gemm_s=evaluation,
      client_import_decrypt_s=decryption,preparation_s=total,correlations_per_s=batch/total,
      noise_bits_min=noise,validation_s=validation,verified_rows=len(ids),verified_all_output_columns=True,
      wire_input_bytes=a.nbytes+zero.nbytes+len(template),wire_output_bytes=len(w)*2*profile.degree*8,
      input_mask_bytes=r.nbytes,output_mask_bytes=wr.nbytes)
    c.close();return r,wr,record
