"""Small trained hybrid decoder for lifecycle tests, not a Qwen checkpoint.

It includes causal convolution, gated DeltaNet, gated full attention with RoPE,
RMS normalization and SwiGLU. The same graph is used for clear and private runs.
"""
from __future__ import annotations
import math,copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .recurrent import scan

CORPUS = ("private inference keeps the prompt on the client. the server stores the model. "
          "fresh masks hide each input. the client checks and decrypts the result. "
          "every used mask is discarded. preparation and inference both take time. "
          "measure the whole process. a fast kernel is not a fast conversation.\n")*16


def quantize(x):
    scale=x.abs().amax(-1,keepdim=True).clamp_min(1e-8)/7
    return (x/scale).round().clamp(-7,7),scale

class Linear(nn.Module):
    def __init__(self,n,m):
        super().__init__();self.weight=nn.Parameter(torch.randn(m,n)/math.sqrt(n))
    def forward(self,x):
        qx,sx=quantize(x);qw,sw=quantize(self.weight)
        # Fake quantization uses ordinary next-token training, not distillation.
        x=x+(qx*sx-x).detach();w=self.weight+(qw*sw-self.weight).detach()
        return F.linear(x,w)

class Block(nn.Module):
    def __init__(self,dim,full):
        super().__init__();self.dim=dim;self.full=full;self.hd=8;self.vheads=2
        self.norm1=nn.RMSNorm(dim,eps=1e-6);self.norm2=nn.RMSNorm(dim,eps=1e-6)
        if full:
            self.proj=Linear(dim,2*dim+16);self.out=Linear(dim,dim)
            self.qnorm=nn.RMSNorm(8,eps=1e-6);self.knorm=nn.RMSNorm(8,eps=1e-6)
        else:
            self.proj=Linear(dim,52);self.out=Linear(16,dim)
            self.conv=nn.Parameter(torch.randn(32,1,4)*.1)
            self.A_log=nn.Parameter(torch.zeros(2));self.dt_bias=nn.Parameter(torch.zeros(2))
            self.gated_norm=nn.RMSNorm(8,eps=1e-6)
        self.up=Linear(dim,4*dim);self.down=Linear(2*dim,dim)
    def forward(self,x,state,linear,prefix):
        residual=x;h=self.norm1(x);z=linear(prefix+'.proj',self.proj,h)
        b,t,_=x.shape;state={} if state is None else state
        if self.full:
            qg,k,v=torch.split(z,[2*self.dim,8,8],-1)
            q,gate=qg.reshape(b,t,self.dim//8,16).chunk(2,-1)
            q=self.qnorm(q);k=self.knorm(k.reshape(b,t,1,8));v=v.reshape(b,t,1,8)
            pos=torch.arange(state.get('pos',0),state.get('pos',0)+t,device=x.device).float()
            # 1/4 rotary dimensions, dimension 8 -> one rotary pair.
            cos=pos.cos()[None,:,None,None];sin=pos.sin()[None,:,None,None]
            def rotate(a):
                a0,a1=a[...,:1],a[...,1:2]
                return torch.cat((a0*cos-a1*sin,a1*cos+a0*sin,a[...,2:]),-1)
            q,k=rotate(q),rotate(k)
            past=state.get('pos',0)
            if 'k' in state:k=torch.cat((state['k'],k),1);v=torch.cat((state['v'],v),1)
            kh=k.repeat_interleave(self.dim//8,2).transpose(1,2);vh=v.repeat_interleave(self.dim//8,2).transpose(1,2)
            score=(q.transpose(1,2)@kh.transpose(-1,-2))/math.sqrt(8)
            mask=torch.arange(k.shape[1],device=x.device)[None,:] <= (past+torch.arange(t,device=x.device))[:,None]
            probs=score.masked_fill(~mask,float('-inf')).softmax(-1)
            a=(probs@vh).transpose(1,2).reshape(b,t,self.dim)*gate.reshape(b,t,self.dim).sigmoid()
            state={'k':k,'v':v,'pos':past+t}
        else:
            qkv,zgate,beta,alpha=torch.split(z,[32,16,2,2],-1)
            prev=state.get('conv',torch.zeros(b,3,32,device=x.device,dtype=x.dtype))
            raw=torch.cat((prev,qkv),1)
            conv=F.conv1d(raw.transpose(1,2),self.conv,groups=32).transpose(1,2)
            q,k,v=torch.split(F.silu(conv),[8,8,16],-1)
            q=F.normalize(q.reshape(b,t,1,8),dim=-1,eps=1e-6).repeat_interleave(2,2)/math.sqrt(8)
            k=F.normalize(k.reshape(b,t,1,8),dim=-1,eps=1e-6).repeat_interleave(2,2)
            v=v.reshape(b,t,2,8)
            decay=(-self.A_log.exp()*F.softplus(alpha+self.dt_bias)).exp()
            initial=state.get('recurrent',torch.zeros(b,2,8,8,device=x.device,dtype=x.dtype))
            a,final,_=scan(initial,q,k,v,decay,beta.sigmoid())
            a=(self.gated_norm(a)*F.silu(zgate.reshape(b,t,2,8))).reshape(b,t,16)
            state={'conv':raw[:,-3:],'recurrent':final}
        x=residual+linear(prefix+'.out',self.out,a)
        u,g=linear(prefix+'.up',self.up,self.norm2(x)).chunk(2,-1)
        x=x+linear(prefix+'.down',self.down,F.silu(u)*g)
        return x,state

class MiniHybrid(nn.Module):
    def __init__(self,vocab,dim=32,layers=4):
        super().__init__();self.vocab=vocab;self.dim=dim
        self.embed=nn.Embedding(vocab,dim)
        self.blocks=nn.ModuleList(Block(dim,i%4==3) for i in range(layers))
        self.norm=nn.RMSNorm(dim,eps=1e-6);self.head=Linear(dim,vocab)
    def forward(self,ids,states=None,executor=None,last_only=False):
        def linear(name,module,x):return module(x) if executor is None else executor(name,x)
        x=self.embed(ids);out=[]
        for i,block in enumerate(self.blocks):
            x,s=block(x,None if states is None else states[i],linear,f'blocks.{i}');out.append(s)
        if last_only:x=x[:,-1:]
        return linear('head',self.head,self.norm(x)),out
    def export(self):
        weights={};scales={}
        for name,module in self.named_modules():
            if isinstance(module,Linear):
                q,s=quantize(module.weight.detach())
                weights[name]=q.to(torch.int8).numpy().copy();scales[name]=s.numpy().copy()
        client=copy.deepcopy(self)
        for module in client.modules():
            if isinstance(module,Linear):module.register_parameter('weight',None)
        return client,weights,scales


def train(steps=160,seed=73):
    torch.manual_seed(seed);torch.set_num_threads(1)
    chars=sorted(set(CORPUS));ids=torch.tensor([chars.index(c) for c in CORPUS],dtype=torch.long)
    model=MiniHybrid(len(chars));opt=torch.optim.AdamW(model.parameters(),lr=.003)
    rng=torch.Generator().manual_seed(seed+1);losses=[]
    for i in range(steps):
        start=torch.randint(0,len(ids)-17,(4,),generator=rng)
        x=torch.stack([ids[s:s+16] for s in start]);y=torch.stack([ids[s+1:s+17] for s in start])
        pred,_=model(x);loss=F.cross_entropy(pred.reshape(-1,len(chars)),y.reshape(-1))
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss))
    model.eval();return model,chars,losses
