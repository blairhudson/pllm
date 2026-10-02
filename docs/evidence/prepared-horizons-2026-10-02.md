# Prepared inventory and public artifact horizons

## Reproduce

From repository root:

```sh
.venv/bin/python scripts/probe_prepared_horizons.py --horizons 1 10 --output /tmp/prepared-horizons.json
```

Requires cached pinned Qwen2.5-0.5B-Instruct commit
`7ae557604adf67be50417f59c2c2f167def9a775`. Default horizons are bounded to one/ten;
100 requires explicit selection. Every completed horizon is checkpointed before
the next begins. The separate retained tiny report covers 1/10/100.

## Measured scope

Two co-located role children, W8A8 prefix-f32, client attention projections,
exact completed-prefill reuse, artifact delivery, two greedy output tokens.
Each policy retains its own public object cache across fresh horizon clients.
One-use material and private state are never shared between clients. Each request
has 34 input tokens. Both policies match output digest and usage in all 22 requests.

| Requests | Idle issued / discarded stage rows | On-demand issued / discarded | Correction push bodies, idle → demand |
| --- | --- | --- | --- |
| 1 | 3,360 / 1,680 | 1,680 / 0 | 55,082,312 → 27,541,156 bytes |
| 10 | 5,040 / 2,928 | 2,112 / 0 | 82,623,468 → 34,762,504 bytes |

Online masked upload/download bodies are identical: 46,217,320 bytes for one
request; 58,622,992 for ten. Both policies record nine exact-prefill hits at ten.
Cold bundle delivery is 189,551,097 bytes; later fresh horizon clients pay
230,556 bytes for authenticated manifest delivery and hit the public cache.
On-demand uses more preparation calls at ten (480 versus 144); control bytes
remain counted, and no preparation occurs during online execution.

Every post-close ledger satisfies issued = claimed + burned + discarded and
reserved = claimed + burned. "Claimed" means consumed cryptographic material,
not successful computation. Unsealed failed issuance is outside this ledger.

These are application-body and row measurements, not full wire, representative
generation quality, independent-provider privacy or whole-response compute-cap
evidence. No 10× fresh-response claim follows. On-demand removes spare waste;
idle refill can reduce next-request wait and stays independently selectable.

Raw reports: `prepared-horizons-qwen25-2026-10-02.json` and
`prepared-horizons-tiny-2026-10-02.json`.
