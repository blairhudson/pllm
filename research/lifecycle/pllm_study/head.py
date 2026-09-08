"""Full 5120 -> 248320 vocabulary projection; no checkpoint quality claim."""
import json,time,resource
from pathlib import Path
import numpy as np
import torch
from .online import OnlineLinear

def main():
    torch.set_num_threads(4);rng=np.random.default_rng(22)
    w=rng.integers(-7,8,(248320,5120),dtype=np.int8)
    start=time.perf_counter();fn=OnlineLinear(w);compile_s=time.perf_counter()-start
    records=[]
    for batch in (1,8,16):
        x=rng.integers(-7,8,(batch,5120),dtype=np.int8)
        d=rng.integers(0,2097169,(batch,5120),dtype=np.uint32);timings={'clear_s':[],'masked_s':[]}
        for repeat in range(3):
            t=time.perf_counter();a=fn(x);timings['clear_s'].append(time.perf_counter()-t)
            t=time.perf_counter();b=fn(d,2097169);timings['masked_s'].append(time.perf_counter()-t)
        chosen=np.linspace(0,len(w)-1,128).astype(int)
        assert np.array_equal(a[:,chosen],x.astype(np.int64)@w[chosen].astype(np.int64).T)
        assert np.array_equal(b[:,chosen],(d.astype(np.int64)@w[chosen].astype(np.int64).T)%2097169)
        r=dict(stage='head_full',batch=batch,input_width=5120,output_width=248320,raw=timings,
               medians={k:float(np.median(v)) for k,v in timings.items()},
               validation='128 output rows, every batch row, independent NumPy integer reference',
               compile_s=compile_s,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        records.append(r);print(json.dumps(r),flush=True)
        Path('results/head-full.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
