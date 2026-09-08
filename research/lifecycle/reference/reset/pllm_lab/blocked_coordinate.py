"""Cache-size tiling for addition-only coordinate packing."""
import numpy as np
import _sealapi_cpp as seal
from .coordinate_bfv import Server

class BlockedServer(Server):
    def __init__(self,profile,weight,block=64):
        super().__init__(profile,weight)
        if block<1: raise ValueError('block must be positive')
        self.block=block

    def evaluate(self,cts,encrypted_zero):
        if len(cts)!=self.weight.shape[1]:raise ValueError('width mismatch')
        accums=[None]*len(self.weight)
        for start in range(0,len(cts),self.block):
            weights=self.weight[:,start:start+self.block]
            partial=object.__new__(Server)
            partial.profile=self.profile
            partial.weight=weights
            partial.last_counts={}
            # Reuse the actual evaluator/context. No extra crypto context setup.
            partial.context=self.context;partial.evaluator=self.evaluator
            terms=partial.evaluate(cts[start:start+self.block],encrypted_zero)
            for j,term in enumerate(terms):
                if accums[j] is None:
                    # Some rows alias an input multiple. Copy by adding zero
                    # before any in-place accumulation to preserve that input.
                    accums[j]=seal.Ciphertext()
                    self.evaluator.add(term,encrypted_zero,accums[j])
                else:self.evaluator.add_inplace(accums[j],term)
        self.last_counts={'rotations':0,'multiplications':0,'input_block':self.block}
        return accums
