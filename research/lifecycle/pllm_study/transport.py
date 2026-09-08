"""Bounded authenticated binary RPC for the isolated lifecycle experiment.

This authenticates transport frames, not computation by a malicious service.
Loopback test endpoint; TLS and production admission are outside this experiment.
"""
import hashlib,hmac,socket,struct
import msgpack

LIMIT=64*1024*1024

class Channel:
    def __init__(self,sock,key,direction):
        self.sock=sock;self.key=key;self.direction=direction;self.sent=0;self.received=0
        self.tx_bytes=0;self.rx_bytes=0
    def _read(self,n):
        chunks=[]
        while n:
            b=self.sock.recv(min(n,1048576))
            if not b:raise EOFError('connection closed')
            chunks.append(b);n-=len(b)
        return b''.join(chunks)
    def send(self,value):
        body=msgpack.packb(value,use_bin_type=True)
        frame=bytes([self.direction])+struct.pack('<Q',self.sent)+body
        if len(frame)>LIMIT:raise ValueError('frame too large')
        tag=hmac.digest(self.key,frame,'sha256');payload=struct.pack('<I',len(frame))+frame+tag
        self.sock.sendall(payload);self.tx_bytes+=len(payload);self.sent+=1
    def receive(self):
        length=struct.unpack('<I',self._read(4))[0]
        if not 9<=length<=LIMIT:raise ValueError('invalid frame length')
        frame=self._read(length);tag=self._read(32);self.rx_bytes+=length+36
        if not hmac.compare_digest(tag,hmac.digest(self.key,frame,'sha256')):raise ValueError('invalid frame authentication')
        if frame[0]!=1-self.direction or struct.unpack_from('<Q',frame,1)[0]!=self.received:raise ValueError('replayed or wrong-direction frame')
        self.received+=1
        return msgpack.unpackb(frame[9:],raw=False,strict_map_key=True)
