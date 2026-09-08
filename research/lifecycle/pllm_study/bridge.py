"""Checked SEAL 4.x serialization bridge. No modification to BFV arithmetic.

Supports expanded, coefficient form BFV ciphertexts with two polynomials and
one coefficient prime. The underlying SEAL loader validates reconstructed
objects. Linux uses anonymous RAM files; other platforms use temporary files.
"""
from __future__ import annotations
import ctypes, ctypes.util, os, struct, sys, tempfile, zlib
from pathlib import Path
import numpy as np
import _sealapi_cpp as seal
H=struct.Struct('<HBBBBHQ')
MAX_OBJECT=16*1024*1024
zstd_name=ctypes.util.find_library('zstd')
if not zstd_name and sys.platform=='darwin':
    for candidate in ('/opt/homebrew/opt/zstd/lib/libzstd.dylib','/usr/local/opt/zstd/lib/libzstd.dylib'):
        if Path(candidate).exists():zstd_name=candidate;break
zstd=ctypes.CDLL(zstd_name or 'libzstd.so.1')
zstd.ZSTD_getFrameContentSize.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
zstd.ZSTD_getFrameContentSize.restype=ctypes.c_ulonglong
zstd.ZSTD_decompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
zstd.ZSTD_decompress.restype=ctypes.c_size_t
zstd.ZSTD_isError.argtypes=[ctypes.c_size_t];zstd.ZSTD_isError.restype=ctypes.c_uint

def wrap(body):return H.pack(0xA15E,16,4,3,0,0,len(body)+16)+body

def unwrap(data):
    if not 16<=len(data)<=MAX_OBJECT:raise ValueError('invalid length')
    magic,hs,major,minor,mode,reserved,size=H.unpack_from(data)
    if (magic,hs,major,reserved,size)!=(0xA15E,16,4,0,len(data)):raise ValueError('invalid SEAL header')
    body=data[16:]
    if mode==0:return body
    if mode==1:
        dec=zlib.decompressobj();out=dec.decompress(body,MAX_OBJECT+1)
        if len(out)>MAX_OBJECT or dec.unconsumed_tail or not dec.eof:raise ValueError('invalid compressed object')
        return out
    if mode!=2:raise ValueError('unsupported compression')
    size=int(zstd.ZSTD_getFrameContentSize(body,len(body)))
    if not 0<=size<=MAX_OBJECT:raise ValueError('invalid expanded length')
    out=ctypes.create_string_buffer(size)
    actual=int(zstd.ZSTD_decompress(out,size,body,len(body)))
    if zstd.ZSTD_isError(actual) or actual!=size:raise ValueError('invalid ZSTD data')
    return out.raw

def array_bytes(a):
    a=np.ascontiguousarray(a,dtype='<u8')
    return wrap(struct.pack('<Q',a.size)+a.tobytes())

def read_array(data,size):
    body=unwrap(data)
    if len(body)!=8+8*size or struct.unpack_from('<Q',body)[0]!=size:raise ValueError('array length mismatch')
    return np.frombuffer(body,dtype='<u8',offset=8)

class Bridge:
    def __init__(self,context):
        self.context=context
        self.fd=os.memfd_create('pllm-seal',os.MFD_CLOEXEC) if hasattr(os,'memfd_create') else None
        self.temp=None if self.fd is not None else tempfile.TemporaryDirectory(prefix='pllm-')
        self.path=Path(f'/proc/self/fd/{self.fd}') if self.fd is not None else Path(self.temp.name)/'object.seal'
        parms=context.first_context_data().parms();mods=parms.coeff_modulus()
        if len(mods)!=1:raise ValueError('one coefficient prime required')
        self.degree=parms.poly_modulus_degree();self.q=int(mods[0].value());self.p=int(parms.plain_modulus().value())
        self.parms=tuple(context.first_context_data().parms_id())
    def close(self):
        if self.fd is not None:os.close(self.fd);self.fd=None
        elif self.temp is not None:self.temp.cleanup()
    def save(self,obj):obj.save(str(self.path));return self.path.read_bytes()
    def ciphertext_array(self,ct):
        body=unwrap(self.save(ct))
        if len(body)<97:raise ValueError('truncated ciphertext')
        parms=struct.unpack_from('<4Q',body);ntt=body[32]
        size,degree,mods=struct.unpack_from('<3Q',body,33);scale,correction=struct.unpack_from('<dQ',body,57)
        if parms!=self.parms or (ntt,size,degree,mods,scale,correction)!=(0,2,self.degree,1,1.,1):raise ValueError('unsupported ciphertext')
        a=read_array(body[73:],2*self.degree)
        if np.any(a>=self.q):raise ValueError('noncanonical residue')
        return a,body[:73]
    def array_ciphertext(self,a,template):
        if a.dtype!=np.uint64 or a.shape!=(2*self.degree,) or len(template)!=73 or np.any(a>=self.q):raise ValueError('invalid coefficients')
        self.path.write_bytes(wrap(template+array_bytes(a)))
        ct=seal.Ciphertext();ct.load(self.context,str(self.path));return ct
    def plaintext(self,a):
        a=np.ascontiguousarray(a,dtype='<u8')
        if a.ndim!=1 or len(a)>self.degree or np.any(a>=self.p):raise ValueError('invalid plaintext')
        self.path.write_bytes(wrap(bytes(32)+struct.pack('<Qd',len(a),1.)+array_bytes(a)))
        pt=seal.Plaintext();pt.load(self.context,str(self.path));return pt
    def plaintext_array(self,pt,count):
        body=unwrap(self.save(pt))
        if body[:32]!=bytes(32):raise ValueError('NTT plaintext unsupported')
        size=struct.unpack_from('<Q',body,32)[0];a=read_array(body[48:],size)
        out=np.zeros(count,dtype=np.int64);out[:min(count,size)]=a[:count];return out
