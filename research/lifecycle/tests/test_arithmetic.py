import numpy as np
import pytest
import torch
from pllm_study.coefficients import CoefficientGEMM
from pllm_study.preparation import Client,Profile,prepare
from pllm_study.online import OnlineLinear,pack_residues,unpack_residues
from pllm_study.bridge import wrap,unwrap

@pytest.mark.parametrize('n,m,b',[(1,1,1),(7,9,16),(64,16,32),(128,3,2)])
def test_integer_linear(n,m,b):
    rng=np.random.default_rng(7);w=rng.integers(-7,8,(m,n),dtype=np.int8)
    x=rng.integers(-7,8,(b,n),dtype=np.int8);d=rng.integers(0,2097169,(b,n),dtype=np.uint32)
    fn=OnlineLinear(w)
    assert np.array_equal(fn(x),x.astype(np.int64)@w.astype(np.int64).T)
    assert np.array_equal(fn(d,2097169),(d.astype(np.int64)@w.astype(np.int64).T)%2097169)

@pytest.mark.parametrize('p',[17,257,65537,2097169])
def test_pack(p):
    a=np.array([[0,1,p-1]],dtype=np.uint32)
    assert np.array_equal(unpack_residues(pack_residues(a,p),a.shape,p),a)
    with pytest.raises(ValueError):pack_residues(np.array([p]),p)
    with pytest.raises(ValueError):unpack_residues(b'bad',(5,),p)

@pytest.mark.parametrize('q',[2**31-1,2**47-115,18014398509404161])
def test_coefficient_arbitrary_precision(q):
    rng=np.random.default_rng(19);a=rng.integers(0,q,(31,17),dtype=np.uint64)
    w=rng.integers(-7,8,(4,31),dtype=np.int8);w[0]=0;zero=rng.integers(0,q,17,dtype=np.uint64)
    result=CoefficientGEMM(a,q,zero).evaluate(w)
    expected=np.array([[sum(int(w[j,k])*int(a[k,i]) for k in range(31))%q for i in range(17)] for j in range(4)],dtype=np.uint64)
    expected[0]=zero;assert np.array_equal(result,expected)

@pytest.mark.parametrize('batch',[1,17,2048])
def test_full_bfv_batch_and_signed_output(batch):
    torch.set_num_threads(1)
    w=np.array([[7,-7,1],[0,0,0],[-7,-7,-7]],dtype=np.int8)
    r,wr,record=prepare(w,Profile(),batch=batch,verify_rows=batch)
    x=np.full(r.shape,7,dtype=np.int64)
    d=(x+r.astype(np.int64))%record['plain_modulus']
    out=OnlineLinear(w)(d,record['plain_modulus'])
    plain=(out.astype(np.int64)-wr.astype(np.int64))%record['plain_modulus']
    plain=np.where(plain>record['plain_modulus']//2,plain-record['plain_modulus'],plain)
    assert np.array_equal(plain,x@w.astype(np.int64).T)
    assert record['noise_bits_min']>0


def test_matches_seal_coefficients():
    import _sealapi_cpp as seal
    c=Client(Profile());r=np.arange(32,dtype=np.uint32).reshape(8,4)
    a,z,template=c.encrypt(r);w=np.array([[7,-3,2,1]],dtype=np.int8)
    actual=CoefficientGEMM(a,c.q if hasattr(c,'q') else c.bridge.q,z).evaluate(w)[0]
    ev=seal.Evaluator(c.context);terms=[]
    for coeff,weight in zip(a,w[0]):
        ct=c.bridge.array_ciphertext(coeff,template)
        for _ in range(abs(int(weight))):
            term=seal.Ciphertext();ev.negate(ct,term) if weight<0 else ev.add(ct,c.bridge.array_ciphertext(z,template),term)
            # Adding fresh encrypted zero changes ciphertext coefficients, so use original for positive weights.
            terms.append(term if weight<0 else ct)
    total=seal.Ciphertext();ev.add_many(terms,total)
    expected,_=c.bridge.ciphertext_array(total)
    assert np.array_equal(actual,expected)
    c.close()


def test_bridge_rejects_malformed():
    with pytest.raises(ValueError):unwrap(b'x')
    data=bytearray(wrap(b'abc'));data[0]=0
    with pytest.raises(ValueError):unwrap(bytes(data))
    c=Client(Profile())
    with pytest.raises(ValueError):c.bridge.plaintext(np.array([2097169],dtype=np.uint64))
    a,z,t=c.encrypt(np.ones((4,2),dtype=np.uint32))
    with pytest.raises(ValueError):c.bridge.array_ciphertext(np.full(a.shape[1],c.bridge.q,dtype=np.uint64),t)
    with pytest.raises(ValueError):c.bridge.array_ciphertext(a[0],t[:-1])
    c.close()


def test_modulus_bound():
    with pytest.raises(ValueError):prepare(np.full((1,1024),7,dtype=np.int8),Profile(modulus=65537),batch=1)


def test_masks_never_reused_property():
    p=257;r=np.arange(p,dtype=np.int64)
    for x in (0,1,128,256):assert np.array_equal(np.sort((x+r)%p),r)
    assert np.array_equal(((7+r)-(31+r))%p,np.full(p,(7-31)%p))

@pytest.mark.parametrize('count',[1,17,2048])
def test_compact_coefficients(count):
    from pllm_study.compact import pack_coefficients,unpack_coefficients
    q=18014398509404161
    a=np.random.default_rng(55).integers(0,q,(3,count),dtype=np.uint64)
    b=pack_coefficients(a,q)
    assert len(b)==a.size*7
    assert np.array_equal(unpack_coefficients(b,a.shape,q),a)
    with pytest.raises(ValueError):unpack_coefficients(b[:-1],a.shape,q)


def test_security_profile_rejects_excess_modulus():
    with pytest.raises(ValueError):Profile(degree=1024,bits=54).context()
