"""Read-only arithmetic inventory; no tensors, native calls, or author execution.

Run: python3 -B paper/methods/fusefss-decoder-feasibility-costs.py
Only --write updates this script's sibling JSON. Counts describe an explicitly
conditional lowering, NOT a selected/admitted FuseFSS decoder or measured costs.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

BASE_COMMIT = "2cb68f75449589217bc0c73e0190fa73378564b1"
FIELDS = (
    "private_material_bytes_both_workers", "directed_native_peer_frames",
    "native_peer_bytes_both_directions", "local_public_table_copy_bytes_both_workers",
    "dcf_keys_per_worker", "dcf_evaluations_per_worker", "dcf_path_levels_per_worker",
    "fresh_triple_pairs", "protected_primitive_calls", "public_scalar_multiplies_both_workers",
)


def cost(**values):
    return Counter(values)


def times(c, n):
    return Counter({k: v * n for k, v in c.items()})


def product():
    # conversions.rs: two 24-byte shares, one (80+16)-byte frame per role.
    return cost(private_material_bytes_both_workers=48,
                directed_native_peer_frames=2, native_peer_bytes_both_directions=192,
                fresh_triple_pairs=1, protected_primitive_calls=1)


def convert(bits, output, shift=0):
    # TR output=bits-shift: full-width correction vanishes modulo output ring.
    # ARS retains it. VectorComparison stores TWO u64 coordinates, even unused.
    widths = ([bits] if ((1 << (bits - shift)) & ((1 << output) - 1)) else [])
    if shift:
        widths.append(shift)
    return cost(private_material_bytes_both_workers=2 * (16 + sum(32 + 34*b for b in widths)),
                directed_native_peer_frames=2, native_peer_bytes_both_directions=176,
                dcf_keys_per_worker=len(widths), dcf_evaluations_per_worker=len(widths),
                dcf_path_levels_per_worker=sum(widths), protected_primitive_calls=1)


def ars():
    return convert(50, 50, 12)


def tr_positive(shift):
    return convert(50, 50-shift, shift) + convert(50-shift, 50)


def comparison(endpoint=False):
    # compile.rs::normalize retains one AND with public whole-ring membership.
    # Interior lt: 2 Query, 3 Bit, 3 Xor, 1 And. Endpoint: 3 Bit,1 Xor,1 And.
    # Includes following BooleanToArithmetic, not a following arithmetic select.
    keys = 0 if endpoint else 2
    return cost(private_material_bytes_both_workers=2*(keys*1324 + 8 + 3*8 + 24 + 24),
                directed_native_peer_frames=6, native_peer_bytes_both_directions=560,
                dcf_keys_per_worker=keys, dcf_evaluations_per_worker=keys,
                dcf_path_levels_per_worker=50*keys, fresh_triple_pairs=2,
                protected_primitive_calls=2)


def lookup(entries):
    # lookup.rs::intervals/resources, full ring50, every chunk evaluated.
    chunks = (entries + 255)//256
    intervals = entries + 2*chunks - 1
    return cost(private_material_bytes_both_workers=2*chunks*(1324+24),
                directed_native_peer_frames=2, native_peer_bytes_both_directions=2*(80+8*chunks),
                local_public_table_copy_bytes_both_workers=2*(24*intervals+8*chunks),
                dcf_keys_per_worker=chunks, dcf_evaluations_per_worker=intervals,
                dcf_path_levels_per_worker=50*intervals, protected_primitive_calls=1,
                public_scalar_multiplies_both_workers=4*(intervals-chunks))


def gelu():
    # Compiled frozen public-boundary affine adapter, NOT generic chunked LUT.
    return cost(private_material_bytes_both_workers=2744,
                directed_native_peer_frames=4, native_peer_bytes_both_directions=368,
                local_public_table_copy_bytes_both_workers=24592,
                dcf_keys_per_worker=1, dcf_evaluations_per_worker=512,
                dcf_path_levels_per_worker=25600, fresh_triple_pairs=1,
                protected_primitive_calls=1, public_scalar_multiplies_both_workers=2044)


def silu():
    return (convert(38,32,6) + convert(32,50) + times(comparison(),2)
            + times(product(),3) + lookup(1024))


def nexp():
    return (convert(8,50) + tr_positive(8) + times(lookup(256),2)
            + product() + tr_positive(12))


def softmax(k):
    reciprocal_entries = 1 << (6 + k.bit_length())
    return (times(comparison()+product(), k-1)
            + times(comparison()+product()+nexp(), k)
            + tr_positive(6) + lookup(reciprocal_entries)
            + times(product()+tr_positive(12),k))


def norm_index():
    # Index: initial *128, seven low-bin scales, final mantissa *64.
    c = cost(public_scalar_multiplies_both_workers=18)
    c += times(comparison()+times(product(),2),49)
    for shift in range(1,43):
        c += convert(50,50-shift,shift) + convert(50-shift,50)
    return c


def norm(width, layer):
    c = times(product(),width) + norm_index()
    if layer:
        c += ars() + cost(public_scalar_multiplies_both_workers=2)
    c += lookup(8192)
    c += times(product()+times(ars(),2)
               + cost(public_scalar_multiplies_both_workers=2), width)
    return c


MODELS = {
    "gpt2-base-dimension-control": dict(layers=12, hidden=768, intermediate=3072,
        heads=12, kv_heads=12, vocab=50257, layer_norm=True, rotary=False,
        metadata_url="https://huggingface.co/openai-community/gpt2/raw/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json"),
    "qwen25-05b-instruct-pinned": dict(layers=24, hidden=896, intermediate=4864,
        heads=14, kv_heads=2, vocab=151936, layer_norm=False, rotary=True,
        metadata_url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/7ae557604adf67be50417f59c2c2f167def9a775/config.json"),
}


def workload(name, prefix, selections):
    m = MODELS[name]
    l,h,i,a,v = (m[k] for k in ("layers","hidden","intermediate","heads","vocab"))
    kv = m["kv_heads"]*(h//a)
    rows = sum(prefix+j for j in range(selections))
    pairs = sum((prefix+j)*(prefix+j+1)//2 for j in range(selections))
    gated = m["rotary"]
    phases = {}
    # EXPLICIT unselected candidate: equality=[token<c+1]-[token<c] in ring50.
    # c=0 uses endpoint descriptor; same frame count, fewer DCFs. No pruning of
    # repeated prefix tokens and no private-row access in public embedding scan.
    equal = times(comparison(),2*v-1) + comparison(endpoint=True)
    phases["embedding_equality_ring50_candidate"] = times(equal,rows)
    matrix_macs = rows*v*h
    phases["embedding_public_matrix"] = cost(public_scalar_multiplies_both_workers=2*matrix_macs)
    body_macs = l*rows*((2*h*h+2*h*kv+3*h*i) if gated else (4*h*h+2*h*i))
    phases["body_public_linear"] = cost(public_scalar_multiplies_both_workers=2*body_macs)
    linear_outputs = l*rows*((3*h+2*kv+2*i) if gated else (5*h+i))
    phases["body_linear_aggregate_floor12_candidate"] = times(ars(),linear_outputs)
    qk = l*h*pairs
    phases["attention_qk_scalar_products_candidate"] = times(product(),qk)
    phases["attention_qk_dot_floor_scale_floor_candidate"] = (
        times(ars(),2*l*a*pairs)
        + cost(public_scalar_multiplies_both_workers=2*l*a*pairs))
    phases["attention_softmax_current_source_schedule"] = Counter()
    reciprocal_histogram = Counter()
    for j in range(selections):
        for k in range(1,prefix+j+1):
            phases["attention_softmax_current_source_schedule"] += times(softmax(k), l*a)
            reciprocal_histogram[1 << (6+k.bit_length())] += l*a
    phases["attention_pv_scalar_products_candidate"] = times(product(),qk)
    phases["attention_pv_aggregate_floor12_candidate"] = times(ars(),l*rows*h)
    phases["body_nonlinear_current_source_schedule"] = times(silu() if gated else gelu(),l*rows*i)
    if gated:
        phases["mlp_gate_up_product_floor_candidate"] = times(product()+ars(),l*rows*i)
        rotary_products = 2*l*rows*(h+kv)
        phases["rotary_two_public_products_two_floors_candidate"] = (
            times(ars(),rotary_products)
            + cost(public_scalar_multiplies_both_workers=2*rotary_products))
    phases["body_norm_current_source_schedule"] = times(norm(h,m["layer_norm"]),2*l*rows)
    phases["final_norm_all_prefix_rows_candidate"] = times(norm(h,m["layer_norm"]),rows)
    phases["last_row_complete_head_public_matrix"] = cost(public_scalar_multiplies_both_workers=2*selections*v*h)
    phases["head_aggregate_floor12_candidate"] = times(ars(),selections*v)
    # Ascending-index sequential scan; strict greater replaces score and index.
    # Requires signed50 nonwrapping differences, otherwise circuit is NOT valid.
    phases["stable_argmax_compare_two_selects_candidate"] = times(comparison()+times(product(),2),selections*(v-1))
    total = sum(phases.values(),Counter())
    unique_weights = (v*h + l*(2*h*h+2*h*kv+3*h*i+3*h+2*kv)+h if gated else
                      v*h+1024*h+l*(4*h*h+2*h*i+9*h+i)+2*h)
    # Decimal link arithmetic is a lower bound only for this candidate ledger.
    # Committed F2 has no dealer codec; this is its uncompressed payload model.
    protected_bytes = total["native_peer_bytes_both_directions"] + total["private_material_bytes_both_workers"]
    return dict(model=name, prefix=prefix, selections=selections,
        admitted=False, circuit_choice="UNKNOWN; named conditional current-scalar candidate only",
        full_prefix_rows=rows, causal_query_key_pairs_per_head=pairs,
        counts=dict(embedding_equalities=rows*v, embedding_macs_per_worker=matrix_macs,
            body_linear_macs_per_worker=body_macs, qk_secret_products=qk,
            pv_secret_products=qk, body_linear_output_floors=linear_outputs,
            nonlinear_scalars=l*rows*i, body_norm_rows=2*l*rows,
            final_norm_rows_baseline=rows, final_norm_rows_after_dead_row_elimination=selections,
            head_macs_per_worker=selections*v*h, head_output_floors=selections*v,
            argmax_comparisons=selections*(v-1)),
        reciprocal_table_calls_by_entries=dict(sorted(reciprocal_histogram.items())),
        phases={k:{f:c[f] for f in FIELDS} for k,c in phases.items()},
        total={f:total[f] for f in FIELDS},
        source_storage=dict(public_parameter_elements=unique_weights,
            public_encoded_u64_payload_per_worker=8*unique_weights,
            public_encoded_u64_two_worker_delivery_if_sent_once=16*unique_weights,
            tied_head_not_duplicated=True, original_source_owner_bytes=None,
            rotary_tables_and_source_import_overlap_bytes=None),
        candidate_full_tensor_payloads_per_worker_not_peak=dict(
            maximum_prefix_rows=prefix+selections-1,
            token_history_u64_bytes=8*(prefix+selections-1),
            one_hidden_tensor_u64_bytes=8*(prefix+selections-1)*h,
            one_intermediate_tensor_u64_bytes=8*(prefix+selections-1)*i,
            one_layer_causal_scores_u64_bytes=8*a*(prefix+selections-1)*(prefix+selections)//2,
            last_row_full_logits_u64_bytes=8*v),
        directed_delivery=dict(dealer_to_worker_a_private_logical_bytes=total["private_material_bytes_both_workers"]//2,
            dealer_to_worker_b_private_logical_bytes=total["private_material_bytes_both_workers"]//2,
            dealer_serialized_frames=None, authentication_and_relay_bytes=None,
            client_to_workers_prompt_share_words_bytes=16*prefix,
            workers_to_client_final_id_share_words_bytes=16*selections,
            client_boundary_codec_frames=None),
        conditional_link_bounds=dict(shared_bottleneck_bytes_peer_plus_uncompressed_keys=protected_bytes,
            seconds_at_100Mbps_shared_link_minimum=protected_bytes*8/100_000_000,
            seconds_at_1Gbps_shared_link_minimum=protected_bytes*8/1_000_000_000,
            aggregate_Gbps_required_within_300_seconds=protected_bytes*8/300/1_000_000_000,
            aggregate_Gbps_required_within_1200_seconds=protected_bytes*8/1200/1_000_000_000),
        dead_final_norm_rows_removable_cost={f:times(norm(h,m["layer_norm"]),rows-selections)[f] for f in FIELDS},
        live_process_peak_rss_bytes=None, live_compiler_metadata_bytes=None,
        elapsed_seconds=None, model_numeric_range_admission=None)


def report():
    # Independent algebraic sanity checks; not native conformance tests.
    assert comparison()["private_material_bytes_both_workers"] == 5456
    assert lookup(65536)["local_public_table_copy_bytes_both_workers"] == 3174352
    assert lookup(131072)["local_public_table_copy_bytes_both_workers"] == 6348752
    assert convert(50,38,12)["private_material_bytes_both_workers"] == 912
    assert ars()["private_material_bytes_both_workers"] == 4376
    templates = {"multiply_or_phi_or_b2a": product(), "comparison50_with_b2a": comparison(),
        "ars50_floor12": ars(), "tr_positive50_floor12": tr_positive(12),
        "gelu_frozen_public_affine": gelu(), "silu": silu(), "nexp": nexp(),
        "layernorm768": norm(768,True), "rmsnorm896_source_guard1_candidate": norm(896,False)}
    return dict(schema="pllm.fusefss.decoder-static-conditional.v1", date="2026-10-20",
        scope="Static current-gate formulas and conditional decoder inventory; no native execution or source tensors.",
        unknown=["selected complete model circuit", "checkpoint numeric ranges", "Qwen frontend/epsilon/RoPE decisions",
            "dealer codecs/authenticated frames", "live full-model compiler/runtime owners", "RSS and execution time"],
        metadata=MODELS,
        formula_baseline_commit=BASE_COMMIT,
        primitive_templates={k:dict(v) for k,v in templates.items()},
        workloads=[workload("gpt2-base-dimension-control",p,t) for p,t in ((128,1),(128,8),(4,2))]
            + [workload("qwen25-05b-instruct-pinned",p,t) for p,t in ((4,2),(64,8))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true",help="write sibling JSON; otherwise stdout")
    args = parser.parse_args()
    text = json.dumps(report(),indent=2,sort_keys=True)+"\n"
    if args.write:
        Path(__file__).with_suffix(".json").write_text(text)
    else:
        print(text,end="")
