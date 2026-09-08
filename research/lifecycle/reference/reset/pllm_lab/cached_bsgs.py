"""Control: compile the old BSGS layout once, and reuse encoded NTT weights."""
import math, time
import numpy as np
import _sealapi_cpp as seal
from pllm.runtime.packed_bfv import PackedBFVLinearFactory


class CachedBSGS:
    def __init__(self,weight,batch):
        self.f=PackedBFVLinearFactory()
        f=self.f;self.w=weight;self.b=batch;self.m,self.n=weight.shape
        self.segment=self.n+self.m-1
        if batch>f.capacity(self.n,self.m):raise ValueError('batch does not fit')
        per_row=f.row_size//self.segment
        self.bases=[(b//per_row)*f.row_size+(b%per_row)*self.segment for b in range(batch)]
        self.baby=f._choose_baby_steps(self.segment)
        self.giants=math.ceil(self.segment/self.baby)
        self.terms=[]
        t=time.perf_counter()
        for giant in range(self.giants):
            shift=giant*self.baby;terms=[]
            for baby in range(self.baby):
                k=shift+baby
                if k>=self.segment:break
                diag=np.zeros(f.slot_count,dtype=np.int64)
                j=np.arange(self.m);i=j+k-(self.m-1);valid=(i>=0)&(i<self.n)
                for base in self.bases:
                    diag[base+j[valid]]=weight[j[valid],i[valid]].astype(np.int64)%f.plain_modulus
                if not np.any(diag):continue
                if shift:
                    diag=np.r_[np.roll(diag[:f.row_size],shift),np.roll(diag[f.row_size:],shift)]
                pt=seal.Plaintext();f.encoder.encode(diag.tolist(),pt)
                ntt=seal.Plaintext();f.evaluator.transform_to_ntt(pt,f.context.first_parms_id(),ntt)
                terms.append((baby,ntt))
            if terms:self.terms.append((shift,terms))
        self.compile_s=time.perf_counter()-t

    def encrypt(self,x):
        f=self.f;flat=np.zeros(f.slot_count,dtype=np.int64)
        for base,row in zip(self.bases,x):flat[base+self.m-1:base+self.m-1+self.n]=row%f.plain_modulus
        pt=seal.Plaintext();f.encoder.encode(flat.tolist(),pt)
        ct=seal.Ciphertext();f.encryptor.encrypt_symmetric(pt,ct);return ct

    def evaluate(self,ct):
        f=self.f;babies=[]
        for j in range(self.baby):
            rot=ct
            if j:
                rot=seal.Ciphertext();f.evaluator.rotate_rows(ct,j,f.galois_keys,rot)
            ntt=seal.Ciphertext();f.evaluator.transform_to_ntt(rot,ntt);babies.append(ntt)
        giants=[]
        for shift,terms in self.terms:
            accum=None
            for idx,weight in terms:
                prod=seal.Ciphertext();f.evaluator.multiply_plain(babies[idx],weight,prod)
                if accum is None:accum=prod
                else:f.evaluator.add_inplace(accum,prod)
            f.evaluator.transform_from_ntt_inplace(accum)
            if shift:f.evaluator.rotate_rows_inplace(accum,shift,f.galois_keys)
            giants.append(accum)
        result=seal.Ciphertext();f.evaluator.add_many(giants,result);return result

    def decrypt(self,ct):
        f=self.f;pt=seal.Plaintext();f.decryptor.decrypt(ct,pt)
        flat=np.asarray(f.encoder.decode_int64(pt),dtype=np.int64)
        return np.stack([flat[base:base+self.m] for base in self.bases])%f.plain_modulus
