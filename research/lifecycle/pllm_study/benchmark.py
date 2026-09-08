import argparse,json,time,resource,platform,os
from pathlib import Path
import numpy as np,torch
from .preparation import Profile,prepare

def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--input',type=int,required=True);p.add_argument('--output',type=int,required=True)
    p.add_argument('--batch',type=int,default=2048);p.add_argument('--rounds',type=int,default=1);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bits',type=int,default=54);p.add_argument('--modulus',type=int,default=2097169);p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--json',type=Path,required=True);a=p.parse_args();torch.set_num_threads(a.threads)
    w=np.random.default_rng(42).integers(-7,8,(a.output,a.input),dtype=np.int8)
    rows=[]
    for i in range(a.rounds):
        r,wr,data=prepare(w,Profile(modulus=a.modulus,bits=a.bits),batch=a.batch,chunk=a.chunk)
        data['round']=i;data['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;data['threads']=a.threads
        rows.append(data);a.json.parent.mkdir(exist_ok=True,parents=True);a.json.write_text(json.dumps(rows,indent=2))
        print(json.dumps(data),flush=True);del r,wr
if __name__=='__main__':main()
