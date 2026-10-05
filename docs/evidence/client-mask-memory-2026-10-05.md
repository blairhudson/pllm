# Seed-backed client mask inventory

The ordinary prepared SDK now retains native SHAKE256 cursors and a one-use row
ledger instead of materializing every stage's input/output masks. It expands only
claimed rows. Domains, seeds, tickets, mask bytes, provider corrections, ring
selection and the wire protocol retain their existing identities. Rust owns XOF
expansion and the burn ledger; PyO3 releases the interpreter lock. Python owns
request/lease orchestration and the independent `hashlib` oracle.

Disjoint concurrent leases can consume rows out of order. Rewinding the XOF uses
a bounded scratch buffer; the ordinary ordered path advances its cursor. Closing
a lease burns its unused ranges without expanding them. Replay, malformed claims,
parent cancellation and verification failures preserve the existing burn boundary.
The explicit Python correctness backend still uses its eager oracle and is not
eligible for native memory admission.

## Isolated measurement

`client-mask-memory-2026-10-05.json` uses the pinned Qwen3-4B semantic geometry:
144 remote stages, 64 prefill rows and seven decode rows. Contexts/seeds are public
fixtures and rings use public worst-case signed-i8 bounds. No checkpoint tensors
or provider processes are loaded; this measures mask handling, not a decoder.

| Measurement | Eager arrays | Native cursors |
| --- | ---: | ---: |
| Retained mask/cursor bytes | 507,764,736 | 224,496 |
| Isolated process peak RSS, MB | 628.16 | 89.60 |
| Inventory construction CPU, s | 0.833 | 0.028 |
| Row claim/hash CPU, s | 0.201 | 0.896 |
| Combined CPU, s | 1.033 | 0.924 |

Every emitted mask byte and ticket matched. The measured mask-process peak falls
**7.01×**, while retained mask storage falls **2,262×**. Combined CPU is 10.5% lower
in this one pair, but expansion moves from inventory construction into stage
execution. This does not establish a whole-response latency improvement. Observed
swap growth was zero. The probe enforces a 2 GiB host-headroom requirement and
uses the existing pressure watchdog.

```bash
uv run --no-sync python scripts/probe_mask_memory.py \
  --config "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c/config.json" \
  --output /tmp/pllm-client-mask-memory.json
```

Use a fresh output path. The cached public configuration is required, not weights.

## Admission and remaining target

`qwen3-4b-lazy-masks-admission-2026-10-05.json` updates the same 64+8 lean prepared
estimate. Client mask allowances fall from 2,031,058,944 to 25,315,312 bytes,
including active-stage conversion buffers and cursor/ledger overhead. Client peak
estimate falls **8.25 → 5.92 GiB** and native topology peak **19.75 → 17.42 GiB**.
The run still rejects before tensor allocation against 4.45 GiB safe headroom.
The host reserve, swap exclusion and allocation margin remain enforced.

The urgent target is at least 10× lower **whole-client measured peak memory**, with
an initial 0.83 GiB estimate budget for this workload. This slice does not meet it.
Boundary weights, import/validation copies and kernel snapshots remain allocation
targets. The semantic executor already releases inputs after their last scheduled
use; correcting its historical all-output workspace allowance alone would be an
estimate correction, not a new memory optimization.

Checked SDK compositions include ordinary and Freivalds-verified preparation,
compressed artifact delivery, generated-prefix reuse and windowed issuance.
No new client body weights, provider trust change or plaintext traffic is added.
