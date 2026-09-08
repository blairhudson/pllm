import socket,secrets,struct,hmac
import numpy as np
import pytest
from pllm_study.transport import Channel


def test_duplex_protocol():
    a,b=socket.socketpair();key=secrets.token_bytes(32)
    x,y=Channel(a,key,0),Channel(b,key,1)
    try:
        x.send({'hello':b'ciphertext'});assert y.receive()=={'hello':b'ciphertext'}
        y.send({'result':1});assert x.receive()=={'result':1}
    finally:a.close();b.close()


def test_replay_rejected():
    a,b=socket.socketpair();key=secrets.token_bytes(32)
    receiver=Channel(b,key,1)
    body=b'\x00'+struct.pack('<Q',0)+b'\x80'
    frame=struct.pack('<I',len(body))+body+hmac.digest(key,body,'sha256')
    try:
        a.sendall(frame);assert receiver.receive()=={}
        a.sendall(frame)
        with pytest.raises(ValueError):receiver.receive()
    finally:a.close();b.close()


def test_authentication_rejected():
    a,b=socket.socketpair();key=secrets.token_bytes(32)
    try:
        Channel(a,secrets.token_bytes(32),0).send({'input':b''})
        with pytest.raises(ValueError):Channel(b,key,1).receive()
    finally:a.close();b.close()

@pytest.mark.parametrize('wrong_epoch',[False,True])
def test_isolated_server_rejects_old_material(wrong_epoch):
    import multiprocessing as mp
    from pllm_study.lifecycle import server,P
    from pllm_study.preparation import Client,Profile
    from pllm_study.online import pack_residues
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe();key=secrets.token_bytes(32)
    weights={'test':np.array([[1,2],[-3,1]],dtype=np.int8)}
    process=ctx.Process(target=server,args=(child,weights,key,0));process.start()
    sock=socket.create_connection(parent.recv());sock.settimeout(3);c=Channel(sock,key,0);hello=c.receive()
    crypto=Client(Profile());r=np.array([[3,4]],dtype=np.uint32);a,z,t=crypto.encrypt(r)
    try:
        epoch=hello['epoch'];c.send(dict(op='prepare',epoch=epoch,stage='test',batch=1,id='batch',
                                      coeff=a.tobytes(),zero=z.tobytes(),q=crypto.bridge.q));c.receive()
        request=dict(op='linear',epoch=epoch,stage='test',ids=[['batch',0]],input=pack_residues(r,P))
        c.send(request);c.receive()
        if wrong_epoch:request['epoch']='different-session'
        c.send(request)
        with pytest.raises(EOFError):c.receive()
        assert 'error' in parent.recv()
    finally:
        sock.close();crypto.close();process.join(5)
        if process.is_alive():process.terminate();process.join()
