import math
import numpy as np
import pytest
import torch
from pllm_study.recurrent import *
from pllm_study.planner import dense_plan,moe_plan

@pytest.mark.parametrize('accepted',range(9))
def test_prefix_replay(accepted):
    torch.manual_seed(7)
    s=torch.randn(1,2,8,8);q=torch.randn(1,8,2,8);k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
    v=torch.randn_like(q);d=torch.full((1,8,2),.9);b=torch.full_like(d,.3)
    y,last,states=scan(s,q,k,v,d,b,save_states=True)
    actual=accepted_prefix_state(s,(q,k,v,d,b),accepted)
    assert torch.equal(actual,s if accepted==0 else states[accepted-1])


def test_reference_numeric_agreement():
    torch.manual_seed(9);s=torch.randn(1,2,8,8,dtype=torch.float64)
    q=torch.randn(1,6,2,8,dtype=torch.float64);k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
    v=torch.randn_like(q);d=torch.full((1,6,2),.8,dtype=torch.float64);b=torch.full_like(d,.4)
    a=scan(s,q,k,v,d,b,step=delta_step);z=scan(s,q,k,v,d,b)
    torch.testing.assert_close(a[0],z[0],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(a[1],z[1],atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('draft,pred,expected,k',[
    ([1,2,3],[1,2,3,4],[1,2,3,4],3),([1,2,3],[5,4,3,2],[5],0),
    ([1,2,3],[1,5,6,7],[1,5],1),([],[3],[3],0)])
def test_greedy(draft,pred,expected,k):assert greedy_accept(draft,pred)==(expected,k)


def test_prompt_lookup():
    assert prompt_candidates([1,2,3,4,1,2],2)==[3,4]
    assert prompt_candidates([1,2,3],4)==[]


def test_speculation_cost_accounts_for_waste():
    assert choose_draft_length(0,1,.1)==0
    assert choose_draft_length(.8,.001,1)==0
    assert choose_draft_length(1,1,.001)>0
    assert expected_accepted(4,.8)<5


def test_model_counts():
    p=dense_plan();assert p['serial_exchanges']==257
    assert p['remote_matrix_parameters']==25621954560
    assert p['mask_inventory_2048_tokens_gib']==48.21875
    assert moe_plan()['expert_compute_multiplier']==32
