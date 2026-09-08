"""Client Gated DeltaNet state, proposal verification, and accepted-prefix replay.

All tensors below are client-private. Replaying arithmetic is not reusing a
network message or a cryptographic mask. Equations follow Gated DeltaNet.
"""
from __future__ import annotations
import math,time,json,argparse
from pathlib import Path
import torch
import numpy as np


def delta_step(state,q,k,v,decay,beta):
    """Reference step. Inputs q,k are already normalized; decay is in (0,1]."""
    state=state*decay[...,None,None]
    correction=(v-(state*k[...,None]).sum(-2))*beta[...,None]
    state=state+k[...,None]*correction[...,None,:]
    return (state*q[...,None]).sum(-2),state


def delta_step_bmm(state,q,k,v,decay,beta):
    """Same operation with matrix kernels; floating summation order may differ."""
    state=state*decay[...,None,None]
    prediction=torch.matmul(k.unsqueeze(-2),state).squeeze(-2)
    correction=(v-prediction)*beta.unsqueeze(-1)
    state=state+k.unsqueeze(-1)*correction.unsqueeze(-2)
    return torch.matmul(q.unsqueeze(-2),state).squeeze(-2),state


def scan(state,q,k,v,decay,beta,step=delta_step_bmm,save_states=False):
    outputs=[];states=[]
    for i in range(q.shape[1]):
        y,state=step(state,q[:,i],k[:,i],v[:,i],decay[:,i],beta[:,i])
        outputs.append(y)
        if save_states:states.append(state.clone())
    return torch.stack(outputs,1),state,states


def accepted_prefix_state(initial,trace,accepted,step=delta_step_bmm):
    q,k,v,decay,beta=trace
    if not 0<=accepted<=q.shape[1]:raise ValueError('invalid accepted prefix')
    if accepted==0:return initial.clone()
    return scan(initial,*[t[:,:accepted] for t in trace],step=step)[1]


def prompt_candidates(history:list[int],limit:int,max_ngram:int=4):
    """Copy continuations of a repeated local suffix. No draft model required."""
    if limit<=0:return []
    for width in range(min(max_ngram,len(history)),0,-1):
        needle=history[-width:]
        for start in range(len(history)-2*width,-1,-1):
            if history[start:start+width]==needle:
                # Never copy from future/unavailable tokens.
                return history[start+width:min(start+width+limit,len(history)-width)]
    return []


def greedy_accept(draft:list[int],target_predictions:list[int]):
    """Accept matching prefix then return one target correction/bonus token."""
    if len(target_predictions)!=len(draft)+1:raise ValueError('one prediction per candidate plus bonus required')
    count=0
    while count<len(draft) and draft[count]==target_predictions[count]:count+=1
    return draft[:count]+[target_predictions[count]],count


def expected_accepted(draft_length:int,accept_probability:float):
    if draft_length<0 or not 0<=accept_probability<=1:raise ValueError('invalid speculation inputs')
    return sum(accept_probability**j for j in range(draft_length+1))


def choose_draft_length(accept_probability,barrier_s,row_s,max_draft=8):
    """Minimise time per delivered token including all evaluated candidate rows.

    row_s must include replenishment and byte costs, not just online GEMM.
    """
    if min(barrier_s,row_s)<0:raise ValueError('negative cost')
    return min(range(max_draft+1),key=lambda k:(barrier_s+(k+1)*row_s)/expected_accepted(k,accept_probability))


def benchmark(path):
    torch.set_num_threads(4);g=torch.Generator().manual_seed(47)
    state=torch.randn((1,48,128,128),generator=g)*.01
    trace=[torch.randn((1,8,48,128),generator=g) for _ in range(3)]
    trace[0]=torch.nn.functional.normalize(trace[0],dim=-1)/math.sqrt(128)
    trace[1]=torch.nn.functional.normalize(trace[1],dim=-1)
    trace += [torch.full((1,8,48),.95),torch.full((1,8,48),.4)]
    records={}
    with torch.inference_mode():
        yr,sr,_=scan(state,*trace,step=delta_step)
        y,s,states=scan(state,*trace,save_states=True)
        error=float((y-yr).abs().max());state_error=float((s-sr).abs().max())
        for name,fn in [('reference_8',lambda:scan(state,*trace,step=delta_step)),
                        ('bmm_8',lambda:scan(state,*trace)),
                        ('snapshots_8',lambda:scan(state,*trace,save_states=True)),
                        ('replay_accepted_4',lambda:accepted_prefix_state(state,trace,4))]:
            raw=[]
            for _ in range(7):
                t=time.perf_counter();fn();raw.append(time.perf_counter()-t)
            records[name]={'raw_s':raw,'median_s':float(np.median(raw))}
        assert torch.equal(accepted_prefix_state(state,trace,4),states[3])
    records.update(output_max_abs_error=error,state_max_abs_error=state_error,
      snapshot_bytes_per_layer=sum(s.numel()*s.element_size() for s in states),
      replay_trace_bytes_per_layer=sum(x.numel()*x.element_size() for x in trace),
      base_state_bytes_per_layer=state.numel()*state.element_size(),
      full_model_delta_layers=48,scope='One real-size Qwen recurrent block, synthetic activations, no learned model')
    path.write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',type=Path,required=True);a=p.parse_args();benchmark(a.json)
