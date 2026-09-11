# Model support

Separate tensor import, graph execution, public bundles, and quality evidence.


| Source or graph | Current source status |
| --- | --- |
| Local float Safetensors, single file or shards | Importer implemented |
| Dense Llama and Mistral decoder graphs | Reference graph implemented |
| Qwen2/Qwen2.5 decoder graph | Reference graph implemented; dashboard default is Qwen2.5-0.5B-Instruct |
| Supported Gemma text layouts | Reference graph implemented |
| Selected MLX affine quantization layouts | Partial importer support |
| vLLM directory containing Hugging Face tensors | Tensors imported; execution is not delegated to vLLM |
| Qwen3.5-27B complete checkpoint generation | Not executed in current validation |
| Sparse expert routing | No private routing implementation |
| GGUF, llama.cpp, or Ollama | Metadata/trusted routing only; no current private graph |

Reading configuration and tensors does not prove complete operator coverage.
Validate a checkpoint in four separate steps: import coverage, clear quantized
execution, private-versus-clear quantized parity, and language quality against the
source model on representative held-out prompts.

## Hub resolution and cache

Remote IDs use Hugging Face credentials. Set `HF_TOKEN` for private or gated
repositories, use stored `hf auth login` state, or pass `--hf-token`. Pin
`--revision` when repeatability matters. `--hf-cache-dir` overrides cache location.

The downloader uses resumable ranged requests and records completed chunks. Set
`PLLM_HF_DOWNLOAD_WORKERS` from 1 through 64 to change concurrency. The default
PLLM model cache is `~/.cache/pllm/models`; `PLLM_HF_MODEL_CACHE` overrides it.
Otherwise `HF_HUB_CACHE` takes precedence over `HF_HOME` as a base location.

A completed cache pins a mutable requested name such as `main` to the resolved
commit. Pass a different explicit revision or remove that model's cache directory
when intentionally refreshing it. Keep the compiled cache separately so unchanged
source files and quantization settings can reuse compiled matrices.

## Public boundary matrices

Public bundle schema 2 includes quantized token lookup and output-head matrices.
The client gathers token rows locally and applies the head only to the final
prefill row, then once per decode step. These vocabulary-sized operations do not
contact either remote service.

For tied embeddings, one vocabulary-by-hidden matrix uses per-token row scales and
is referenced by both boundaries. This lookup is not numerically identical to the
old schema 1 independently quantized transpose; output-head quantization remains
unchanged. Untied embeddings and model-specific per-layer token tables stay
separate.

Count bundle network bytes, verified disk cache, and client memory when evaluating
this tradeoff. Confidential-weight bundles do not expose these matrices and retain
remote token lookup/output projection, so their measurements are not comparable to
the public path.
