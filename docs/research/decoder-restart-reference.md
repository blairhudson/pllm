# Checkpoint-backed decoder restart gate

The installed state target's `decoder_restart` operation extends storage mechanics
to a real, bounded compiled decoder. The exact W8A8 `prefix_f32` prepared baseline
is explicit. Source headers preflight at most 16 MiB of float32 weights before
loading values, token fixtures stop at 128, native continuation admits geometry
under 64 MiB, and authenticated checkpoint frames stay within 8 MiB.

The trusted issuer executes a prefill and one selected token, then checkpoints
complete float32 KV, final logits, executed token history and the next greedy
token. The pending token is not included in saved KV or executed-history length.
AEAD binds the compiled source/body, numeric composition, continuation schedule
and trusted revision. Transactional publication does not replace an existing
revision; interrupted writes leave the prior trusted head usable.

A new interpreter independently loads and validates the checkpoint source,
authenticates the complete frame, validates every KV extent and finite value,
checks the pending token against saved logits, and constructs fresh process-local
provenance. Two continuation steps must preserve every logit and final KV word,
position and executed-history commitment against uninterrupted execution.
The local prepared arithmetic reference uses original correction domains and a
fresh bounded one-use native mask cursor per stage. It agrees with the independent
clear W8A8 stage executor on generated Qwen2, Qwen3 and Llama prefill. Material
attempts before/after restoration are distinct. Persisted frames contain no old
inventory, seed, credential or process-local seal.

## Scope and replay

```sh
python -m benchmarks.research.decoder_restart --processes 3 \
  --output target/decoder-restart.json
```

The generated-fixture study uses 8/32/128 input tokens, one executed generated
token and two resumed steps, with three fresh issuer/client pairs per case. Reports
retain original bootstrap, publication bytes/CPU, restarted-client loading and
restoration, combined sample CPU, separate process peaks and fresh arithmetic
bodies. These local arithmetic bodies exclude real provider transport and full
wire. Original checkpoint distribution and an application-facing SDK conversation
are not exercised.

The issuer retains the AEAD key and latest head outside untrusted storage. Durable
key custody, cross-process writer arbitration, a persistent anti-rollback service,
sampling modes beyond greedy and verified-lineage restoration remain absent.
Both simulated prepared roles share each client process; this proves continuity,
not independent-provider privacy. Python-owned secret buffers retain ordinary
allocation lifetimes. Official large checkpoints reject this reference's explicit
source bound. A restart-heavy full SDK comparison must charge first bootstrap,
publication, fresh preparation and restored requests before any tenfold claim.
