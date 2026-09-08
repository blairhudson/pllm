"""Direct BFV dense layers with ordinary coefficient convolution and row tiling.

This is a packing experiment related to existing rotation-free HE linear
protocols, NOT a new cryptosystem. The weights are unchanged signed integers.
Unused decrypted coefficients are not hidden from the client, so the backend
only accepts PUBLIC weights and has no model-confidentiality claim.
"""
from __future__ import annotations
import numpy as np
import _sealapi_cpp as seal
from .coordinate_bfv import Profile, plaintext


class CoefficientServer:
    def __init__(self, profile: Profile, weight: np.ndarray):
        if weight.ndim != 2 or weight.dtype.kind not in 'iu':
            raise ValueError('integer weight matrix required')
        if np.any(weight < -7) or np.any(weight > 7):
            raise ValueError('W4 weights required')
        self.profile=profile
        self.context=profile.context()
        self.evaluator=seal.Evaluator(self.context)
        self.n=weight.shape[1]
        self.m=weight.shape[0]
        if self.n>profile.degree:
            raise ValueError('input must fit in one polynomial')
        self.rows_per_ct=profile.degree//self.n
        self.compiled=[]
        for j in range(0,self.m,self.rows_per_ct):
            rows=weight[j:j+self.rows_per_ct]
            if not np.any(rows):
                self.compiled.append((None,len(rows)))
                continue
            pt=plaintext(rows.reshape(-1),profile.modulus)
            encoded=seal.Plaintext()
            self.evaluator.transform_to_ntt(pt,self.context.first_parms_id(),encoded)
            self.compiled.append((encoded,len(rows)))

    def evaluate(self,encrypted_input,encrypted_zero):
        input_ntt=seal.Ciphertext()
        self.evaluator.transform_to_ntt(encrypted_input,input_ntt)
        outputs=[]
        for w,rows in self.compiled:
            if w is None:
                outputs.append(encrypted_zero)
                continue
            product=seal.Ciphertext()
            self.evaluator.multiply_plain(input_ntt,w,product)
            self.evaluator.transform_from_ntt_inplace(product)
            outputs.append(product)
        return outputs


def encrypt_vector(client,x):
    x=np.asarray(x)
    ct=seal.Ciphertext()
    client.encryptor.encrypt_symmetric(plaintext(x[::-1],client.profile.modulus),ct)
    zero=seal.Ciphertext()
    client.encryptor.encrypt_symmetric(plaintext([0],client.profile.modulus),zero)
    return ct,zero


def decrypt_outputs(client,server,cts):
    values=[];noise=[]
    for ct,(_,rows) in zip(cts,server.compiled,strict=True):
        pt=seal.Plaintext();client.decryptor.decrypt(ct,pt)
        for j in range(rows):
            at=(j+1)*server.n-1
            values.append(int(pt[at]) if at<pt.coeff_count() else 0)
        noise.append(client.decryptor.invariant_noise_budget(ct))
    return np.asarray(values,dtype=np.int64), min(noise)
