"""Exact modular GEMM on BFV coefficients, with input digits prepared once."""
import numpy as np
import torch

class CoefficientGEMM:
    def __init__(self,a,q,zero=None):
        if a.dtype!=np.uint64 or a.ndim!=2 or not 2<=q<2**54 or np.any(a>=q):raise ValueError('invalid coefficient matrix')
        self.q=q;self.shape=a.shape;self.zero=zero
        self.digits=[]
        # A byte view avoids full-size uint64 shift temporaries. XOR maps an
        # unsigned digit b to the signed int8 representation of b - 128.
        a=np.ascontiguousarray(a,dtype='<u8')
        byteview=a.view(np.uint8).reshape(*a.shape,8)
        for index in reversed(range((q.bit_length()+7)//8)):
            limb=np.bitwise_xor(byteview[:,:,index],np.uint8(128)).view(np.int8)
            self.digits.append(torch.from_numpy(np.ascontiguousarray(limb)))
    def evaluate(self,w):
        if w.dtype!=np.int8 or w.ndim!=2 or w.shape[1]!=self.shape[0] or np.any(w < -7) or np.any(w > 7):raise ValueError('W4 matrix required')
        if w.shape[1]*7*128>=2**31:raise ValueError('int32 product bound exceeded')
        matrix=torch.from_numpy(np.ascontiguousarray(w))
        correction=128*w.astype(np.int64).sum(1)[:,None]
        result=np.zeros((len(w),self.shape[1]),dtype=np.int64)
        for digit in self.digits:
            dot=torch._int_mm(matrix,digit).numpy().astype(np.int64)
            result=(result*256+dot+correction)%self.q
        out=result.astype(np.uint64)
        rows=~np.any(w,axis=1)
        if rows.any():
            if self.zero is None or self.zero.shape!=(self.shape[1],):raise ValueError('encrypted zero required')
            out[rows]=self.zero
        return out
