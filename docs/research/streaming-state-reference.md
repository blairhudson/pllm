# Streaming full-KV storage gate

`BenchmarkTarget.state(..., storage="sealed_stream")` uses the same bounded,
source/plan/layout-bound synthetic KV fixture as `sealed_local`. AES-GCM frames
are unchanged. Encryption uses a 1 MiB scratch buffer; decryption writes into
one record allocation and verifies its tag and trusted frame digest before
returning a read-only float32 view. Failed authentication clears that allocation.

Each generation publishes only after every record and manifest is complete.
Failed writes retain the previous trusted revision and remove pending files.
The caller retains its key and manifest head separately from untrusted storage.
This is not a persistent anti-rollback service. Successful NumPy allocations,
caller views and Python encryption-key objects have ordinary Python lifetimes;
this reference does not claim complete memory zeroization.

Admission retains the existing 64 MiB record and 256 MiB total payload bounds.
The attention consumer keeps complete layer records, with a conservative
three-record codec allocation bound plus bounded ciphertext scratch. Attention
workspace, initialization, caller owners and filesystem cache remain separate.

## Validation

One-shot and streaming codecs read each other's frames bit-for-bit. Exact attention
matches at short and multi-chunk lengths. Corruption, changed trusted revisions,
wrong contexts, truncated frames and interrupted publication reject. The installed
target also executes restoration in a new process, with keys on stdin and separate
child CPU/RSS accounting. This synthetic restart does not authorize decoder state.

Replay the matched, three-process-per-cell study:

```sh
python -m benchmarks.research.state_working_set --processes 3 \
  --output target/state-working-set.json
```

It binds pinned Qwen2.5 configuration-derived KV shapes at 256, 2,048 and 8,192
tokens to resident, one-shot and streaming controls. Values are public synthetic
fixtures. Publication CPU/disk, target attention/restore CPU and process peaks
remain separate. The five-process `--tokens 16 --processes 1` smoke checks the
executor; it establishes no full-decoder memory or performance improvement.

The next gate is whether bounded streaming improves the previous sealed-layout
CPU penalty enough to justify a compiler-bound layer-swapped state owner. Such an
owner still needs real prefill/decode provenance, exact logits and KV, append and
failure lifetimes, and full-response measurement. The SDK state target enables
that decision without activating a serving component.
