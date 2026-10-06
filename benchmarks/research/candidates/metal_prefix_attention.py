from benchmarks.research.combinations import build


def experiment():
    return build("search-metal-prefix-attention", kernel="metal", capsule=True,
                 roles=("qkv_projection", "attention_output"), prefix_layers=2)
