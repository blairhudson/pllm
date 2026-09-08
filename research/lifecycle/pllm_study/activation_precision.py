"""Test eight bit activations without changing the W4 coefficient kernel."""
import time,json
from pathlib import Path
import numpy as np
import torch
from .preparation import prepare,Profile
from .online import OnlineLinear


def main():
    torch.set_num_threads(4);n=17408
    w=np.stack([np.full(n,7,dtype=np.int8),np.full(n,-7,dtype=np.int8)])
    p=33554467
    r,wr,record=prepare(w,Profile(modulus=p),batch=16,verify_rows=16)
    x=np.full(r.shape,127,dtype=np.int64)
    d=(r.astype(np.int64)+x)%p
    masked=OnlineLinear(w)(d,p)
    y=(masked.astype(np.int64)-wr.astype(np.int64))%p;y=np.where(y>p//2,y-p,y)
    expected=x@w.astype(np.int64).T
    assert np.array_equal(y,expected)
    record.update(activation_bits=8,weight_bits=4,activation_extreme=127,
      signed_extrema=[int(expected.min()),int(expected.max())],exact_signed_extrema=True,
      scope='Arithmetic and parameter test only; no Qwen quality evaluation')
    Path('results/activation8.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
