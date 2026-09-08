"""Public execution plan derived from official Qwen text configurations.

These counts do not assert compatibility of the historical PLLM importer.
"""
from dataclasses import dataclass,asdict
import json
from pathlib import Path

@dataclass(frozen=True)
class Stage:
    name:str
    input_width:int
    output_width:int
    count:int

DENSE_STAGES=[Stage('delta_input',5120,16480,48),Stage('attention_input',5120,14336,16),
              Stage('mixer_output',6144,5120,64),Stage('gate_up',5120,34816,64),
              Stage('down',17408,5120,64),Stage('head',5120,248320,1)]

def dense_plan():
    n=sum(s.input_width*s.count for s in DENSE_STAGES)
    m=sum(s.output_width*s.count for s in DENSE_STAGES)
    params=sum(s.input_width*s.output_width*s.count for s in DENSE_STAGES)
    return dict(model='Qwen/Qwen3.5-27B',scope='Text graph, public local embedding; remote dense projections and head',
      stages=[asdict(s) for s in DENSE_STAGES],serial_exchanges=sum(s.count for s in DENSE_STAGES),
      input_values_per_token=n,output_values_per_token=m,remote_matrix_parameters=params,
      online_payload_bytes_24bit=3*(n+m),preparation_raw_bytes_per_full_batch_token=16*(n+m),
      mask_inventory_bytes_per_token_uint32=4*(n+m),mask_inventory_2048_tokens_gib=2048*4*(n+m)/2**30,
      output_only_inventory_2048_tokens_gib=2048*4*m/2**30,
      local_embedding_ideal_int4_mib=248320*5120/2/2**20,
      local_embedding_bf16_mib=248320*5120*2/2**20,
      recurrent_state_fp32_mib=48*48*128*128*4/2**20,
      conv_state_fp32_mib=48*4*10240*4/2**20,
      attention_kv_bf16_bytes_per_context_token=16*2*4*256*2,
      sources=['https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json',
               'https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py'])


def moe_plan():
    # All-expert comparison does not reveal routing, but requires every expert.
    # Expert gate/up and down count exclude shared expert/router/attention.
    return dict(model='Qwen/Qwen3.5-35B-A3B',layers=40,experts=256,selected=8,hidden=2048,
      expert_width=512,active_expert_parameters_per_token=40*8*3*2048*512,
      oblivious_all_expert_parameters_per_token=40*256*3*2048*512,
      expert_compute_multiplier=32,local_expert_weights_ideal_int4_gib=40*256*3*2048*512/2/2**30,
      warning='Sending selected expert identifiers reveals a private-input-dependent route; no private dispatch protocol is supplied',
      source='https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json')

if __name__=='__main__':
    target=Path(__file__).resolve().parents[1]/'results/model-plan.json'
    target.write_text(json.dumps({'dense':dense_plan(),'moe':moe_plan()},indent=2))
    print(target.read_text())
