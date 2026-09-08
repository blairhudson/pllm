import numpy as np
import pytest
from pllm.runtime.stage_protocol import MaskedStageRequest, DirectFHEStageRequest, BlindedStageRequest
from pllm.runtime.linear_integrity import create_linear_check_key,check_linear_result
from pllm.runtime.secure_random import FieldRandom,uniform_residues


def test_masked_scales_remain_local():
    scale=np.asarray([0.19273641],dtype=np.float32)
    r=MaskedStageRequest('m','s','c',np.asarray([1,2],dtype=np.uint32),scale,65537,24)
    assert scale.tobytes() not in r.pack()
    np.testing.assert_array_equal(MaskedStageRequest.unpack(r.pack()).activation_scales,[1.0])


def test_direct_scales_remain_local():
    scale=np.asarray([0.19273641],dtype=np.float32)
    r=DirectFHEStageRequest('m','s','ctx',(1,2),scale,(b'ciphertext',))
    assert scale.tobytes() not in r.pack()
    np.testing.assert_array_equal(DirectFHEStageRequest.unpack(r.pack()).activation_scales,[1.0])


def test_blinded_scales_remain_local():
    scale=np.asarray([0.19273641],dtype=np.float32)
    r=BlindedStageRequest('m','s','owner',('c',),np.asarray([[1,2]],dtype=np.uint32),scale,65537,24)
    assert scale.tobytes() not in r.pack()
    np.testing.assert_array_equal(BlindedStageRequest.unpack(r.pack()).activation_scales,[1.0])


def test_default_integrity_challenge_not_seed_one():
    w=np.arange(64*64,dtype=np.int64).reshape(64,64)%7
    key=create_linear_check_key(w,modulus=65537)
    old=create_linear_check_key(w,modulus=65537,seed=1)
    assert not np.array_equal(key.challenge,old.challenge)
    x=np.arange(64,dtype=np.int64)[None,:]
    assert check_linear_result(x,x@w.T%65537,key).valid


@pytest.mark.parametrize('p',[17,257,65537,786433,2**32])
def test_os_residue_bounds(p):
    x=uniform_residues(p,(100,30))
    assert x.dtype==np.uint32 and x.shape==(100,30)
    assert int(x.max())<p
    assert not np.array_equal(x,uniform_residues(p,(100,30)))


def test_rejection_boundary(monkeypatch):
    import pllm.runtime._native_support as native_support
    import pllm.runtime.secure_random as module
    calls=iter([np.asarray([2**32-1,256,255],dtype='<u4').tobytes(),np.asarray([1],dtype='<u4').tobytes()])
    monkeypatch.setattr(native_support, 'extension', lambda: None)
    monkeypatch.setattr(module.secrets,'token_bytes',lambda n: next(calls))
    # For modulus 257, only uint32 maximum is outside the largest full multiple.
    np.testing.assert_array_equal(uniform_residues(257,(3,)),[256,255,1])


def test_invalid_rng_bounds():
    with pytest.raises(ValueError): uniform_residues(1,(2,))
    with pytest.raises(ValueError): uniform_residues(257,(-1,))
    with pytest.raises(ValueError): FieldRandom().integers(1,10,size=3)
