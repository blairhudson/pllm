# Private LLM Inference: from preparation to decoding

This is a reproducible research workspace for PLLM, not a new production release.
It completes the previous coefficient matrix experiment through actual BFV
ciphertext reconstruction and client decryption. It measures complete dense
stages at the dimensions of Qwen3.5 27B and executes a trained small hybrid
decoder across separate client and server processes.

The paper is in `paper/Private-LLM-Inference.pdf`. Read `REPORT.md` for results.
Every numerical table is derived from the JSON records in `results/`.

## Run with UV

```bash
uv sync --group test
uv run python -m pytest -q
uv run python -m pllm_study.planner
uv run python -m pllm_study.lifecycle --json results/demo.json
uv run python -m pllm_study.lifecycle --rtt-ms 10 --draft 4 --json results/demo-speculative.json
```

The serialization bridge needs Zstandard. On macOS, install it with
`brew install zstd`; the bridge recognizes both Apple Silicon and Intel
Homebrew locations.

The lifecycle commands train a small model on the short corpus in the source,
start a separate server process, prepare actual BFV correlations from an empty
inventory, generate 96 tokens, replenish exhausted stage inventories, and check
that the generated tokens equal those from the clear W4A4 graph. They do not
load a Qwen checkpoint. The short corpus is a functional fixture, not a language
quality evaluation dataset.

The prototype uses a local authenticated binary channel. Injected delay is an
application delay per online exchange, not a measurement of a wide area link.
The production PLLM Responses facade is not replaced or revalidated here.

## Cryptographic stage benchmarks

```bash
uv run python -m pllm_study.benchmark --input 768 --output 256 --rounds 3 --json results/preparation.json
uv run python -m pllm_study.benchmark --input 5120 --output 34816 --rounds 2 --json results/qwen-expansion.json
uv run python -m pllm_study.online --json results/online-new.json
uv run python -m pllm_study.head
uv run python -m pllm_study.compact --json results/compact-new.json
uv run python -m pllm_study.recurrent --json results/recurrent-new.json
uv run python -m pllm_study.summarize
```

The expansion benchmark is a real matrix of the target dimensions, with
synthetic public W4 values. It does not establish the quality of quantized Qwen
weights. The full output head benchmark allocates about 1.27 GB of int8 weights;
allow at least 4 GiB of system memory. Run benchmarks sequentially to avoid
contention. Do not run the full model planner's projections as though they were
measurements of a loaded 27B model.

## What changed

* A checked SEAL serialization bridge completes coefficient GEMM through
  actual decryption. Secret values remain on the client.
* Input digit extraction uses a byte view rather than large integer shift
  temporaries. Digits are prepared once for all output chunks.
* Online matrix data is compiled once, outside the inference loop.
* Standard seeded BFV serialization reduces input traffic. A lossless
  seven-byte format reduces the returned coefficient payload.
* A greedy prompt lookup experiment verifies several proposed tokens in one
  batch. Rejected rows still spend fresh correlations. The client reconstructs
  accepted recurrent state from saved local intermediates without another
  private query.
* The cost model includes client work, preparation traffic, inventory size,
  startup, rejected proposals, and every serial model stage.

## Evidence boundary

The principal cryptographic profile is BFV with polynomial degree 2048, one
54-bit coefficient prime, and plaintext modulus 2097169. SEAL accepts it at its
TC128 setting. This is library parameter validation, not an independent
cryptographic review. No rotations, bootstraps, or multiplication of two
ciphertexts occur in the coordinate preparation evaluator.

This protocol protects inputs against a server that follows the protocol.
Weights are public. It does not protect a proprietary model against chosen
layer queries and does not enforce correct computation by a malicious provider.
Transport authentication does not change that assumption. Timing, shapes,
request counts and preparation schedules remain visible.

All activation scales stay local. Fresh masks use operating system randomness
with rejection sampling. Both processes burn correlations before an online
reply. Preparation is discarded when the demonstration session ends. Resistance
to restoring an entire virtual machine snapshot is not implemented.

The public embedding remains local in the 27B cost plan. A BF16 embedding alone
is approximately 2.37 GiB; an ideal int4 table is 606.25 MiB before scale metadata.
The plan does not silently substitute a free private embedding lookup.

## Files

* `pllm_study/`: implementation, cost planner and experiment entry points.
* `tests/`: arithmetic, serialization, replay, transport and evidence checks.
* `results/`: raw samples, whole lifecycle traces and derived tables.
* `reference/`: preserved prior coordinate implementation and provenance.
* `paper/`: one two-column manuscript and its source.
* `docs/`: lifecycle and capacity notes using Material for MkDocs.

The prior public SDK and CLI remain the application interface. This workspace
is invoked through Python modules so that experimental Qwen scheduling is not
advertised as a supported Hugging Face importer.
