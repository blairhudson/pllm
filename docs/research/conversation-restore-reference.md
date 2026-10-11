# Prepared conversation recovery

`Conversation.checkpoint` and `Conversation.restore` connect the installed
full-KV conversation handle to authenticated local storage. The ordinary SDK
still owns prepared execution and fresh inventory. `AsyncConversation` exposes
the same operations with cancellation drainage.

## Contract and ownership

- Select bounded `ClientPrefixReuse` and ordinary `prefix_f32` full-KV execution.
  Native continuation admission checks the complete compiled state geometry
  against the handle's `max_state_bytes` before storage or response execution.
  Retained/windowed state, tiled execution and Freivalds reject.
- Supply a fresh directory and a caller-held 32-byte encryption key. AES-GCM
  pages reuse the transactional context-graph storage contract with a distinct
  conversation domain. Source/composition/state bindings and executed history
  accompany the logits and every active KV row.
- The separately authenticated metadata binds the numeric-state head, complete
  response, message history and pending selection. Metadata is preflighted under
  2 MiB, 32 levels and 32,768 visited values. A pending sampled token is retained
  as pending, rather than falsely becoming an executed KV row.
- `ConversationCheckpoint` is an immutable, serializable trusted head. The
  caller retains its freshness and key custody. Disk content cannot select its
  own latest revision. Old caller-retained heads remain valid recoverable states.
- Saving leaves the live chain usable. All files are synchronized before a head
  returns. Restoration authenticates every page and the native state contract
  before publishing a fresh client-local response ID. Failure exposes no partial
  state; async cancellation waits for storage to retire and closes the chain.
- Credentials, masks, correction tickets and provider leases are absent from
  the checkpoint. The next send uses ordinary fresh inventory. Storage reads
  execute no provider matrix stage.

The state limit prices one retained KV/logit snapshot. Snapshot overlap,
plaintext metadata, encryption buffers, source/bundle import, tokenizer, prefix
cache and immutable checkpoint files remain additional owners. Backup or
cross-device transfer would incur separate traffic. Each publication uses a new
directory; the application owns eventual file retirement, including unreferenced
ciphertext left by interrupted publication.

## Validation

Generated Qwen2, Qwen3 and Llama checkpoints preserve complete logits, active KV,
executed history and pending selection across same-process and fresh-interpreter
restoration. Both prepared providers run in separate children. Restoration itself
issues zero correction rows; subsequent continuation uses freshly issued material
with zero plaintext prompt/token-ID bytes sent.

Regressions cover wrong keys, corrupted numeric pages, failed metadata publication,
empty-handle admission, unsupported numeric state, metadata allocation bounds and
async cancellation while restored state is in flight. The public
`examples/conversation_checkpoint.py` runs against the installed wheel.

## Measurement driver

`scripts/benchmark_conversation_restore.py` compares three turns through:

1. One live RAM conversation.
2. A fresh interpreter replaying the complete message history each turn.
3. A fresh interpreter restoring the last encrypted checkpoint each turn.

Each cell owns new separate provider processes and a cold public object cache.
That verified cache survives client restarts within the cell. Shared native
100/40 Mbps caps cover all service streams. The cohorts alternate order, hold the
exclusive frontier lock, freeze the installed build and driver, and enforce the
existing topology/headroom/swap admission plus a 512 MiB launcher/storage allowance.

The driver checks salted commitments to every final-turn logit and active KV bit,
history, pending selection and output. Reports retain directed application bodies,
all-link TCP-stream counts, client lifetime CPU/peak RSS, provider birth CPU and
sampled RSS, first-text delay, N−1 text-delta throughput, publication/restoration
CPU and logical encrypted reads/writes. Client CPU includes imports and parity
hashes. The launcher owns the relay and resource sampler; its CPU is separate and
its independent peak RSS is unknown. Provider startup RSS peaks, filesystem cache,
physical disk I/O, checkpoint distribution and physical wire remain unmeasured.

Keys, trusted heads, prompts, message history and provider credentials pass only
through private launcher pipes. Evidence stores salted commitments and counts.
RAM hits are the lower-work control: persistence has no inherent traffic advantage
when the existing live conversation already retains the same exact state.

## Matched Qwen2.5 result

`docs/evidence/research-conversation-restore-sdk.json` records three alternating
replicates per mode on pinned Qwen2.5-0.5B W8A8: **50/76/100 input tokens and eight
outputs per turn**. Nine cells use 21 fresh client interpreters and 18 provider
children. All 27 turn states match the RAM control in complete logits, active KV,
executed history, pending selection and output, covering **17,760,384 checked
float32 values**. The host guard observes zero swap growth.

Each entry below is the median of three complete cells. Decimal MB are used;
client CPU sums its interpreters, while client peak is the maximum individual
interpreter lifetime peak within each cell, never the sum of those peaks.

| Complete three-turn cell | Live RAM | Restart and replay | Restart and restore |
| --- | ---: | ---: | ---: |
| All-link TCP-stream bytes | 373.07 MB | 684.31 MB | **373.95 MB** |
| Covered application bodies | 370.07 MB | 680.82 MB | 370.60 MB |
| Online application bodies | 152.60 MB | 345.84 MB | 152.60 MB |
| Fresh correction rows | 10,125 | 23,043 | 10,125 |
| Client lifetime CPU | 26.17 s | 33.91 s | **52.96 s** |
| Maximum client peak RSS | 145.06 MB | 153.11 MB | 148.95 MB |
| Preparation CPU from birth | 20.97 s | 29.79 s | 20.02 s |
| Preparation sampled peak RSS | 159.43 MB | 195.54 MB | 161.19 MB |
| Inference CPU from birth | 30.74 s | 43.80 s | 30.19 s |
| Launcher window CPU | 7.02 s | 10.49 s | 6.73 s |
| Sum of charged CPU windows | 84.27 s | 117.76 s | 109.27 s |
| Complete cell elapsed time | 131.51 s | 196.98 s | **157.79 s** |

Restoration reduces all-link streams **45.35%**, covered bodies **45.57%** and
complete elapsed time **19.90%** against restart-and-replay. Client CPU increases
**56.17%**, while Preparation CPU falls **32.78%**. The sum of charged CPU windows
falls **7.22%**; it includes client lifetime, provider birth and launcher window
costs, not unmeasured launcher work before the cell. Medians of individual CPU
owners need not add to the median of each cell's sum.

Two encrypted publications write **4.684 MB** in total; restoration reads the
same logical bytes. Median publication/restoration CPU is **12.12/14.58 seconds**,
including their SDK source/state binding work. Both remain charged above. The
local checkpoint bytes are additional disk owners, not network savings. Moving
them to backup or another device would add its own traffic.

| Turn metric | Live RAM | Restart and replay | Restart and restore |
| --- | ---: | ---: | ---: |
| First text, turns 1 / 2 / 3 | 62.38 / 16.57 / 16.27 s | 62.54 / 43.64 / 52.56 s | 61.60 / 16.66 / 15.77 s |
| N−1 decode throughput, turns 1 / 2 / 3 | 2.92 / 2.55 / 2.81 tok/s | 2.83 / 2.86 / 2.73 tok/s | 2.69 / 2.65 / 2.82 tok/s |

First-text timing starts at `chat.send`; restoration precedes it and is included
in complete-cell elapsed time and CPU. Provider startup precedes RSS sampling.
These are local shared-100/40-Mbps samples, without added Internet RTT, full
physical-wire accounting or independent operators. The retained source checkpoint
and pre-positioned tokenizer index have separate distribution costs.

**Admit explicit recovery, not automatic persistence.** Restore nearly reaches
the live RAM traffic floor but remains slower and uses more client CPU. It earns
the restart/replay saving only when exact state would otherwise be lost; it does
not improve an existing live conversation or establish a general tenfold gain.

## Replay

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= -q \
  tests/test_conversation_checkpoint.py tests/test_conversation.py
target/frontier-sdk-env/bin/python examples/conversation_checkpoint.py
target/frontier-sdk-env/bin/python scripts/benchmark_conversation_restore.py \
  --processes 3 --output target/conversation-restore-replay.json
```

The benchmark defaults to the pinned indexed-tokenizer Experiment in
`target/prepared-frontier-artifacts/indexed.json`. `--experiment` accepts another
canonical Experiment; `--prompts` accepts a private JSON list of three strings.
Each run creates a new private cohort salt. Different invocations are separate
cohorts and cannot be pooled into a comparison.
