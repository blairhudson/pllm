"""Derived results are rebuilt from measured records and public stage counts."""
import json,statistics,csv,platform,os
from pathlib import Path
from .planner import dense_plan,moe_plan,DENSE_STAGES
R=Path(__file__).resolve().parents[1]/'results'

def load(name):return json.loads((R/name).read_text())
def median_rows(name):
    rows=load(name)
    return {k:(v if all(r[k] == v for r in rows) else statistics.median(r[k] for r in rows))
            for k,v in rows[0].items()
            if isinstance(v,(int,float)) and not isinstance(v,bool)}

def main():
    plan=dense_plan();stages={
      'delta_input':median_rows('optimized-delta-input.json'),
      'attention_input':median_rows('optimized-attention-input.json'),
      'mixer_output':median_rows('optimized-mixer-output.json'),
      'gate_up':median_rows('optimized-gate-up.json'),
      'down':median_rows('optimized-down.json')}
    head=median_rows('optimized-head-tile.json')
    # One shared input; output costs estimated from a 1024-row tile. Do not
    # multiply encryption or digit preparation by the vocabulary tile count.
    ratio=248320/1024
    head['server_gemm_s']*=ratio;head['client_import_decrypt_s']*=ratio
    head['preparation_s']=sum(head[k] for k in ('random_s','client_encrypt_export_s','server_digits_s','server_gemm_s','client_import_decrypt_s'))
    head['correlations_per_s']=2048/head['preparation_s']
    head['wire_output_bytes']=248320*2*2048*8
    head['output_mask_bytes']=248320*2048*4
    head['measurement_scope']='Projection of output work from 1024 measured rows; not a complete head preparation measurement'
    head['measured_output_width']=1024
    head['projected_output_width']=248320
    head['peak_rss_scope']='Measured tile only, not projected head'
    stages['head']=head
    for name, stage in stages.items():
        stage['correlations_per_s']=2048/stage['preparation_s']
    prep=sum(stages[s.name]['preparation_s']/2048*s.count for s in DENSE_STAGES)
    server=sum((stages[s.name]['server_gemm_s']+stages[s.name]['server_digits_s'])/2048*s.count for s in DENSE_STAGES)
    client=prep-server
    online=load('online-compiled.json')['results'];fullhead=load('head-full.json')
    rates=[]
    for batch in (1,8,16):
        data={r['stage']:r for r in online if r['batch']==batch}
        data['head']=next(r for r in fullhead if r['batch']==batch)
        clear=sum(data[s.name]['medians']['clear_s']*s.count for s in DENSE_STAGES)/batch
        masked=sum(data[s.name]['medians']['masked_s']*s.count for s in DENSE_STAGES)/batch
        rates.append(dict(batch=batch,clear_linear_s_per_row=clear,masked_linear_s_per_row=masked,
                          clear_linear_rows_per_s=1/clear,masked_linear_rows_per_s=1/masked))
    rawprep=plan['preparation_raw_bytes_per_full_batch_token'];on=plan['online_payload_bytes_24bit']
    compact=load('compact.json');input_per_coordinate=statistics.median(r['input_bytes']/768/2048 for r in compact)
    comprep=input_per_coordinate*plan['input_values_per_token']+14*plan['output_values_per_token']
    summaries=[]
    for name in ('lifecycle.json','lifecycle-rtt1.json','lifecycle-speculative.json','lifecycle-speculative-rtt1.json',
                 'lifecycle-rtt10.json','lifecycle-speculative-rtt10.json'):
        d=load(name);summaries.append(dict(file=name,draft=d.get('draft_length',0),delay_ms=d['rtt_injected_ms'],
          tokens=d['generated_tokens'],private_s=d['private']['wall_s'],private_tps=d['generated_tokens']/d['private']['wall_s'],
          first_token_s=d['private']['first_token_s'],clear_s=d['clear']['wall_s'],prep_s=d['preparation_seconds'],
          online_rpc_s=d['online_rpc_seconds'],prepared=d['prepared_correlations'],used=d['consumed_correlations'],
          unused=d['unused_correlations'],calls=d['audit']['operations']['linear'],exact_tokens=d['exact_tokens']))
    controls=load('matched-control.json');new=median_rows('optimized-768-256.json')
    old=statistics.median(x['total_s'] for x in controls[0]['samples'])
    comparison=dict(shape=[768,256],batch=2048,old_s=old,new_s=new['preparation_s'],speedup=old/new['preparation_s'])
    # Bounds, not performance forecasts: assume perfect compute overlap.
    links=[]
    for gbps in (.1,1,10,100):
        bandwidth=gbps*1e9/8
        for name,prebytes,input_factor,out_factor in [('raw',rawprep,16,16),('compact',comprep,input_per_coordinate,14)]:
            up=(input_factor+3)*plan['input_values_per_token'];down=(out_factor+3)*plan['output_values_per_token']
            links.append(dict(gbps=gbps,format=name,aggregate_link_tps_ceiling=bandwidth/(prebytes+on),
                              full_duplex_tps_ceiling=bandwidth/max(up,down),online_only_tps_ceiling=bandwidth/on))
    result=dict(environment=dict(python=platform.python_version(),platform=platform.platform(),cpu_count=os.cpu_count(),
      cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),memory_limit=Path('/sys/fs/cgroup/memory.max').read_text().strip(),gpu=False),
      plan=plan,moe=moe_plan(),matched_preparation_comparison=comparison,preparation_stage_medians=stages,
      full_model_preparation_projection=dict(total_cpu_s_per_token=prep,server_s_per_token=server,
        client_s_per_token=client,steady_token_sets_per_s=1/prep,batch_2048_compute_s=prep*2048,
        head_extrapolation='1024 output rows scaled to 248320; input work counted once',
        all_model_weights_loaded=False),linear_projection=rates,compact_preparation_bytes_per_token=comprep,
      network_bounds=links,lifecycle=summaries,
      scope='No full 27B checkpoint generation; exact-size public synthetic stages and a separate trained small decoder')
    (R/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    with (R/'lifecycle-summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=summaries[0].keys());w.writeheader();w.writerows(summaries)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
