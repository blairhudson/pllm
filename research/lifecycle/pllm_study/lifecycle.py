"""A trained small decoder through isolated client and server processes.

All preparation is actual BFV. No trusted dealer or prefilled inventory is used.
The threat model is an honest-but-curious server with public model weights.
"""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp,secrets,socket,time,traceback
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from .mini_model import MiniHybrid,train,quantize
from .preparation import Client,Profile
from .coefficients import CoefficientGEMM
from .randomness import uniform_residues
from .online import OnlineLinear,integer_linear,pack_residues,unpack_residues
from .transport import Channel
from .recurrent import prompt_candidates,greedy_accept

P=2097169


def server(pipe,weights,key,rtt):
    torch.set_num_threads(1)
    compiled={name:OnlineLinear(w) for name,w in weights.items()}
    epoch=secrets.token_hex(16);used=set();batches={};audit={'operations':{},'fields':set(),'server_gemm_s':0.,'preparation_gemm_s':0.}
    sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(1);pipe.send(sock.getsockname())
    conn,_=sock.accept();conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);c=Channel(conn,key,1)
    c.send({'epoch':epoch,'profile':{'degree':2048,'modulus':P,'bits':54}})
    try:
        while True:
            msg=c.receive();op=msg['op'];audit['fields'].update(msg)
            if op=='close':
                audit['fields']=sorted(audit['fields']);audit['used_correlations']=len(used)
                audit['upload_bytes']=c.rx_bytes;audit['download_bytes']=c.tx_bytes
                c.send(audit);break
            if msg['epoch']!=epoch:raise ValueError('wrong session epoch')
            stage=msg['stage'];w=weights[stage];n=w.shape[1]
            audit['operations'][op]=audit['operations'].get(op,0)+1
            if op=='prepare':
                if set(msg)!=set(('op','epoch','stage','batch','id','coeff','zero','q')):raise ValueError('unexpected preparation metadata')
                batch=msg['batch'];identity=msg['id']
                if not 1<=batch<=2048 or identity in batches:raise ValueError('invalid preparation batch')
                # Public context is fixed by the client/server execution plan.
                q=Profile().context().first_context_data().parms().coeff_modulus()[0].value()
                if msg['q']!=q or len(msg['coeff'])!=n*4096*8 or len(msg['zero'])!=4096*8:raise ValueError('invalid coefficient payload')
                a=np.frombuffer(msg['coeff'],dtype='<u8').reshape(n,4096)
                z=np.frombuffer(msg['zero'],dtype='<u8')
                t=time.perf_counter();out=CoefficientGEMM(a,q,z).evaluate(w);audit['preparation_gemm_s']+=time.perf_counter()-t
                batches[identity]=(stage,batch)
                c.send({'coeff':out.astype('<u8',copy=False).tobytes()})
            elif op=='linear':
                if set(msg)!=set(('op','epoch','stage','ids','input')):raise ValueError('unexpected online metadata')
                identities=[tuple(x) for x in msg['ids']]
                if not identities or len(identities)!=len(set(identities)):raise ValueError('duplicate correlations')
                for bid,index in identities:
                    if bid not in batches or batches[bid][0]!=stage or not 0<=index<batches[bid][1] or (bid,index) in used:
                        raise ValueError('invalid or consumed correlation')
                used.update(identities)  # Burn before work and before reply.
                d=unpack_residues(msg['input'],(len(identities),n),P)
                t=time.perf_counter();out=compiled[stage](d,P);audit['server_gemm_s']+=time.perf_counter()-t
                if rtt:time.sleep(rtt)
                c.send({'output':pack_residues(out,P)})
            else:raise ValueError('unknown operation')
    except Exception:
        pipe.send({'error':traceback.format_exc()})
    finally:conn.close();sock.close()


class Remote:
    def __init__(self,address,key,scales,weights_shapes,batch=64):
        sock=socket.create_connection(address);sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        self.channel=Channel(sock,key,0);hello=self.channel.receive();self.epoch=hello['epoch']
        self.crypto=Client(Profile(**hello['profile']));self.q=self.crypto.bridge.q
        self.scales=scales;self.shapes=weights_shapes;self.batch=batch;self.pools={}
        self.preparation_s=0.;self.online_s=0.;self.prepared=0;self.consumed=0;self.events=[]
    def prepare(self,name):
        t=time.perf_counter();m,n=self.shapes[name];r=uniform_residues(P,(self.batch,n))
        a,z,template=self.crypto.encrypt(r);identity=secrets.token_hex(16)
        self.channel.send(dict(op='prepare',epoch=self.epoch,stage=name,batch=self.batch,id=identity,
                               coeff=a.astype('<u8',copy=False).tobytes(),zero=z.astype('<u8',copy=False).tobytes(),q=self.q))
        response=self.channel.receive();out=np.frombuffer(response['coeff'],dtype='<u8').reshape(m,4096)
        wr,noise=self.crypto.decrypt(out,template,self.batch)
        if noise<=0:raise ValueError('insufficient decryption margin')
        self.prepared+=self.batch;self.pools[name]=[identity,0,r,wr]
        elapsed=time.perf_counter()-t;self.preparation_s+=elapsed
        self.events.append({'kind':'preparation','stage':name,'seconds':elapsed,'count':self.batch,'noise_bits':noise})
    def __call__(self,name,x):
        qx,sx=quantize(x.detach());shape=qx.shape;v=qx.reshape(-1,shape[-1]).numpy().astype(np.int64)
        masks=[];outputs=[];ids=[]
        for row in range(len(v)):
            if name not in self.pools or self.pools[name][1]==self.batch:self.prepare(name)
            identity,index,r,wr=self.pools[name]
            self.pools[name][1]+=1 # Permanently reserved even if transport fails.
            masks.append(r[index]);outputs.append(wr[index]);ids.append([identity,index]);self.consumed+=1
        t=time.perf_counter();d=(v+np.stack(masks).astype(np.int64))%P
        self.channel.send(dict(op='linear',epoch=self.epoch,stage=name,ids=ids,input=pack_residues(d,P)))
        answer=self.channel.receive();u=unpack_residues(answer['output'],(len(v),self.shapes[name][0]),P)
        y=(u.astype(np.int64)-np.stack(outputs).astype(np.int64))%P
        y=np.where(y>P//2,y-P,y)
        self.online_s+=time.perf_counter()-t
        # Scales never leave the client.
        return (torch.from_numpy(y.astype(np.float32)).reshape(*shape[:-1],y.shape[-1])*
                sx*torch.from_numpy(self.scales[name].T))
    def close(self):
        self.channel.send({'op':'close'});audit=self.channel.receive();self.channel.sock.close();self.crypto.close();return audit


def clear_executor(weights,scales):
    compiled={name:OnlineLinear(w) for name,w in weights.items()}
    def execute(name,x):
        q,s=quantize(x.detach());shape=q.shape
        y=compiled[name](q.reshape(-1,shape[-1]).numpy().astype(np.int8))
        return torch.from_numpy(y.astype(np.float32)).reshape(*shape[:-1],y.shape[-1])*s*torch.from_numpy(scales[name].T)
    return execute


def generate(model,prompt,count,executor):
    started=time.perf_counter();logits,state=model(torch.tensor([prompt]),executor=executor,last_only=True)
    tokens=[];intervals=[];last=time.perf_counter();first=None
    for i in range(count):
        token=int(logits[0,-1].argmax());tokens.append(token)
        now=time.perf_counter();intervals.append(now-last);last=now
        if first is None:first=now-started
        if i+1<count:logits,state=model(torch.tensor([[token]]),states=state,executor=executor,last_only=True)
    return tokens,{'wall_s':time.perf_counter()-started,'first_token_s':first,'emission_intervals_s':intervals},logits


def generate_speculative(model,prompt,count,executor,draft_length):
    """Greedy target verification with local replay of accepted private state.

    Every proposed row consumes fresh correlations, including rejected rows.
    Fixed candidate count per block avoids transmitting the local match length.
    Timing and number of verification blocks remain observable metadata.
    """
    start=time.perf_counter()
    logits,state=model(torch.tensor([prompt]),executor=executor,last_only=True)
    pending=int(logits[0,-1].argmax());tokens=[pending];first=time.perf_counter()-start
    stats={'verification_blocks':0,'draft_rows':0,'accepted_draft_rows':0,'replay_seconds':0.,'saved_replay_rpc':0}
    while len(tokens)<count:
        candidate=prompt_candidates(prompt+tokens,draft_length)
        candidate=(candidate+[0]*draft_length)[:draft_length]
        inputs=[pending]+candidate;cached={}
        def trace(name,x):
            y=executor(name,x);cached[name]=y;return y
        target,next_state=model(torch.tensor([inputs]),states=state,executor=trace)
        predictions=target[0].argmax(-1).tolist();accepted,k=greedy_accept(candidate,predictions)
        keep=1+k
        if keep==len(inputs):state=next_state
        else:
            t=time.perf_counter()
            def replay(name,x):return cached[name][:,:keep]
            _,state=model(torch.tensor([inputs[:keep]]),states=state,executor=replay)
            stats['replay_seconds']+=time.perf_counter()-t
            stats['saved_replay_rpc']+=len(cached)
        stats['verification_blocks']+=1;stats['draft_rows']+=draft_length;stats['accepted_draft_rows']+=k
        tokens.extend(accepted);pending=accepted[-1]
    # Replaying only saved client intermediates does not submit another query.
    return tokens[:count],{'wall_s':time.perf_counter()-start,'first_token_s':first,**stats},target


def main():
    p=argparse.ArgumentParser();p.add_argument('--json',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=96);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--draft',type=int,default=0)
    p.add_argument('--train-steps',type=int,default=160);p.add_argument('--rtt-ms',type=float,default=0.)
    args=p.parse_args();torch.set_num_threads(1)
    t=time.perf_counter();model,chars,losses=train(args.train_steps);train_s=time.perf_counter()-t
    checkpoint=args.json.parent/'mini-trained.safetensors';save_file(model.state_dict(),str(checkpoint))
    client,weights,scales=model.export();del model
    shapes={k:v.shape for k,v in weights.items()};prompt=[chars.index(c) for c in 'private inference ']
    with torch.inference_mode():
        t=time.perf_counter();clear,cm,cl=generate(client,prompt,args.tokens,clear_executor(weights,scales))
        ctx=mp.get_context('spawn');parent,child=ctx.Pipe();key=secrets.token_bytes(32)
        process=ctx.Process(target=server,args=(child,weights,key,args.rtt_ms/1000));process.start();address=parent.recv()
        remote=Remote(address,key,scales,shapes,args.batch)
        private,pm,pl=(generate_speculative(client,prompt,args.tokens,remote,args.draft) if args.draft else
                       generate(client,prompt,args.tokens,remote))
        assert private==clear,'private tokens differ from clear W4A4 target'
        if not args.draft:assert torch.equal(pl,cl),'last logits differ'
        audit=remote.close();process.join(10)
        if process.exitcode!=0:raise RuntimeError('server did not finish cleanly')
    result=dict(scope='Trained small 4-block hybrid decoder, not a Qwen checkpoint or quality benchmark',
      training_steps=args.train_steps,training_seconds=train_s,initial_training_loss=losses[0],final_training_loss=losses[-1],
      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      generated_tokens=args.tokens,prompt_tokens=len(prompt),batch=args.batch,rtt_injected_ms=args.rtt_ms,
      clear=cm,private=pm,preparation_seconds=remote.preparation_s,online_rpc_seconds=remote.online_s,
      prepared_correlations=remote.prepared,consumed_correlations=remote.consumed,
      unused_correlations=remote.prepared-remote.consumed,
      preparation_events=remote.events,audit=audit,exact_tokens=True,exact_final_logits=(True if not args.draft else None),draft_length=args.draft,
      generated_text=''.join(chars[t] for t in private),client_model_state_bytes=sum(t.numel()*t.element_size() for t in client.state_dict().values()),
      server_weight_bytes=sum(v.nbytes for v in weights.values()),
      all_preparation_included=True,empty_initial_inventory=True,server_process_has_he_secret=False)
    args.json.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
