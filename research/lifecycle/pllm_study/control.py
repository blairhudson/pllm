"""Rerun the previous coordinate-addition evaluator under matched parameters."""
import sys,os,json,time
from pathlib import Path
import numpy as np,torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'reference/reset'))
from pllm_lab.coordinate_bfv import Client,Server,dumps,loads
from pllm_study.preparation import Profile
from pllm_study.randomness import uniform_residues

def main():
    torch.set_num_threads(4);records=[];p=Profile()
    for n,m in [(768,256),(1536,512)]:
        w=np.random.default_rng(42).integers(-7,8,(m,n),dtype=np.int8)
        client=Client(p);server=Server(p,w)
        samples=[]
        for _ in range(3):
            t=time.perf_counter();r=uniform_residues(p.modulus,(2048,n)).astype(np.int64)
            inputs,zero=client.encrypt(r)
            payload=[dumps(ct) for ct in inputs]+[dumps(zero)]
            inp=[loads(b,server.context) for b in payload]
            start=time.perf_counter();out=server.evaluate(inp[:-1],inp[-1]);eval_s=time.perf_counter()-start
            response=[dumps(ct) for ct in out]
            result,noise=client.decrypt([loads(b,client.context) for b in response],2048)
            elapsed=time.perf_counter()-t
            # Independent whole-batch output verification, excluded from timing.
            assert np.array_equal(result,(r@w.astype(np.int64).T)%p.modulus)
            samples.append(dict(total_s=elapsed,server_s=eval_s,noise_min=noise,rate=2048/elapsed))
        records.append(dict(shape=[n,m],profile=dict(degree=p.degree,modulus=p.modulus,bits=p.bits),samples=samples))
        (root/'results/matched-control.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)
if __name__=='__main__':main()
