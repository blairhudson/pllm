"""Standard seeded SEAL encryption plus lossless seven-byte coefficients."""
from __future__ import annotations
import json,time,argparse
from pathlib import Path
import numpy as np
import torch
import _sealapi_cpp as seal
from .preparation import Client,Profile
from .bridge import Bridge
from .coefficients import CoefficientGEMM
from .randomness import uniform_residues


def pack_coefficients(a,q):
    if a.dtype!=np.uint64 or np.any(a>=q) or q>=2**56:raise ValueError('invalid coefficients')
    return np.ascontiguousarray(a.astype('<u8',copy=False).reshape(-1).view(np.uint8).reshape(-1,8)[:,:7]).tobytes()


def unpack_coefficients(payload,shape,q):
    size=int(np.prod(shape))
    if len(payload)!=size*7:raise ValueError('invalid packed length')
    out=np.zeros((size,8),dtype=np.uint8);out[:,:7]=np.frombuffer(payload,dtype=np.uint8).reshape(size,7)
    result=out.reshape(-1).view('<u8').reshape(shape)
    if np.any(result>=q):raise ValueError('noncanonical coefficient')
    return result


def run(n=768,m=256,batch=2048):
    torch.set_num_threads(4);profile=Profile();client=Client(profile);remote=Bridge(profile.context())
    weights=np.random.default_rng(13).integers(-7,8,(m,n),dtype=np.int8)
    masks=uniform_residues(profile.modulus,(batch,n));payload=[]
    t=time.perf_counter()
    for i in range(n):
        serial=client.encryptor.encrypt_symmetric(client.bridge.plaintext(masks[:,i]))
        payload.append(client.bridge.save(serial))
    encryption=time.perf_counter()-t
    t=time.perf_counter();a=np.empty((n,4096),dtype=np.uint64)
    for i,data in enumerate(payload):
        remote.path.write_bytes(data);ct=seal.Ciphertext();ct.load(remote.context,str(remote.path))
        a[i],template=remote.ciphertext_array(ct)
    expansion=time.perf_counter()-t
    # No zero rows in this fixture, so the factory does not need encrypted zero.
    t=time.perf_counter();out=CoefficientGEMM(a,remote.q).evaluate(weights);evaluation=time.perf_counter()-t
    t=time.perf_counter();packed=pack_coefficients(out,remote.q);packing=time.perf_counter()-t
    t=time.perf_counter();restored=unpack_coefficients(packed,out.shape,remote.q);unpacking=time.perf_counter()-t
    assert np.array_equal(restored,out)
    t=time.perf_counter();wr,noise=client.decrypt(restored,template,batch);decryption=time.perf_counter()-t
    assert np.array_equal(wr,(masks.astype(np.int64)@weights.astype(np.int64).T)%profile.modulus)
    rawbytes=(n+m)*4096*8;wire=sum(map(len,payload))+len(packed)
    result=dict(shape=[n,m],batch=batch,client_seeded_encrypt_s=encryption,
      server_expand_s=expansion,server_gemm_s=evaluation,server_pack_s=packing,
      client_unpack_s=unpacking,client_decrypt_s=decryption,noise_min=noise,exact_all_outputs=True,
      raw_wire_bytes=rawbytes,compact_wire_bytes=wire,input_bytes=sum(map(len,payload)),output_bytes=len(packed),
      wire_reduction=1-wire/rawbytes,total_s=sum((encryption,expansion,evaluation,packing,unpacking,decryption)),
      scope='No physical network; standard seeded BFV inputs, lossless coefficient byte packing')
    client.close();remote.close();return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--json',type=Path,required=True);a=p.parse_args()
    r=[run() for _ in range(3)];a.json.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
