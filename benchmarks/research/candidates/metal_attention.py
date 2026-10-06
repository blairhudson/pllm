from benchmarks.research.combinations import build


def experiment():
    return build("search-metal-attention", kernel="metal", roles=("qkv_projection", "attention_output"), prefix_layers=2)
