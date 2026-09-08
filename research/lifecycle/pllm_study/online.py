"""Exact W4 integer linear kernels and public-width transport packing."""
from __future__ import annotations
import time, argparse, json, resource
from pathlib import Path
import numpy as np
import torch


class OnlineLinear:
    """Compile invariant weight data once, outside the request path."""
    def __init__(self,w):
        if w.dtype!=np.int8 or w.ndim!=2 or np.any(w < -7) or np.any(w > 7):
            raise ValueError('W4 integer matrix required')
        if w.shape[1]*7*128>=2**31:raise ValueError('int32 overflow')
        self.weight=np.ascontiguousarray(w)
        self.matrix=torch.from_numpy(self.weight)
        self.correction=128*self.weight.sum(1,dtype=np.int64)[:,None]
    def __call__(self,x,modulus=None):
        if x.ndim!=2 or x.shape[1]!=self.weight.shape[1]:raise ValueError('invalid shape')
        if modulus is None:
            if np.any(x < -128) or np.any(x > 127):raise ValueError('int8 inputs required')
            return torch._int_mm(self.matrix,torch.from_numpy(np.ascontiguousarray(x.T,dtype=np.int8))).numpy().T.copy()
        if not 2<=modulus<=2**31 or np.any(x < 0) or np.any(x >= modulus):raise ValueError('invalid residues')
        result=np.zeros((len(self.weight),len(x)),dtype=np.int64)
        xt=x.T.astype(np.uint32)
        for shift in reversed(range(0,(modulus-1).bit_length(),8)):
            digit=np.ascontiguousarray((((xt>>shift)&255).astype(np.int16)-128).astype(np.int8))
            dot=torch._int_mm(self.matrix,torch.from_numpy(digit)).numpy().astype(np.int64)
            result=(result*256+dot+self.correction)%modulus
        return result.T.astype(np.uint32)


def integer_linear(w,x,modulus=None):
    """Convenience reference; production users should retain OnlineLinear."""
    return OnlineLinear(w)(x,modulus)


def pack_residues(x: np.ndarray, p: int) -> bytes:
    if x.dtype.kind not in 'iu' or np.any(x < 0) or np.any(x >= p):
        raise ValueError('noncanonical residues')
    width=((p-1).bit_length()+7)//8
    if not 1<=width<=4: raise ValueError('unsupported modulus')
    raw=x.astype('<u4').reshape(-1).view(np.uint8).reshape(-1,4)
    return np.ascontiguousarray(raw[:,:width]).tobytes()


def unpack_residues(payload: bytes, shape: tuple[int,...], p: int) -> np.ndarray:
    width=((p-1).bit_length()+7)//8;count=int(np.prod(shape))
    if len(payload)!=count*width:raise ValueError('payload length mismatch')
    padded=np.zeros((count,4),dtype=np.uint8)
    padded[:,:width]=np.frombuffer(payload,dtype=np.uint8).reshape(count,width)
    out=padded.reshape(-1).view('<u4').reshape(shape)
    if np.any(out>=p):raise ValueError('noncanonical payload')
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--json',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    rng=np.random.default_rng(311)
    cases=[('delta_input',5120,16480),('attention_input',5120,14336),
           ('mixer_output',6144,5120),('gate_up',5120,34816),('down',17408,5120),
           ('head_tile',5120,1024)]
    results=[];p=2097169
    for name,n,m in cases:
        w=rng.integers(-7,8,(m,n),dtype=np.int8)
        compiled=OnlineLinear(w)
        for b in (1,8,16):
            x=rng.integers(-7,8,(b,n),dtype=np.int8)
            # Random synthetic residues are a latency fixture, not live masks.
            d=rng.integers(0,p,(b,n),dtype=np.uint32)
            expected=(d.astype(np.int64)@w.astype(np.int64).T)%p
            raw={key:[] for key in ('clear_s','masked_s','codec_s')}
            for repeat in range(4):
                t=time.perf_counter();clear=compiled(x);raw['clear_s'].append(time.perf_counter()-t)
                t=time.perf_counter();masked=compiled(d,p);raw['masked_s'].append(time.perf_counter()-t)
                t=time.perf_counter();inp=pack_residues(d,p);out=pack_residues(masked,p)
                assert np.array_equal(unpack_residues(out,masked.shape,p),masked)
                raw['codec_s'].append(time.perf_counter()-t)
            assert np.array_equal(masked,expected)
            assert np.array_equal(clear,x.astype(np.int32)@w.astype(np.int32).T)
            r=dict(stage=name,input_width=n,output_width=m,batch=b,modulus=p,
                   threads=args.threads,raw=raw,medians={k:float(np.median(v)) for k,v in raw.items()},
                   input_bytes=len(inp),output_bytes=len(out),exact=True,
                   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            results.append(r);print(json.dumps(r),flush=True)
            args.json.write_text(json.dumps({'scope':'Synthetic exact target matrix shapes; no trained checkpoint weights',
                                             'results':results},indent=2))
        del w

if __name__=='__main__':main()
