# Trusted-local preparation placement screen — 2026-10-01

## Decision

**Conditional trust/locality candidate; reject as a 10× all-link network method.**
Moving preparation to a user-controlled always-on workstation or home appliance
moves the small client/preparation seed-and-ack link onto LAN or loopback. It
does **not** move the large preparation/inference correction upload off WAN.
The separate-appliance all-link body sum is exactly unchanged. Even an
unimplemented IPC replacement removes only 21,278 counted network bytes for the
baseline or 10,662 for the attention-placement variant.

This screen reclassifies archived real loopback bodies. No hosts were deployed,
no endpoint ownership was checked, and no latency/availability improvement was
measured. Heavy online linear inference remains remote. Public weights are
reused locally after their initial distribution; fresh correction computation
and upload recur for every inventory.

## Protocol first: algebra, freshness, trust

For each public quantized linear stage, with public integer matrix
`W ∈ Z_(2^k)^(out × in)` and an input row `x`:

1. Client and Prep hold a private stage root. Existing SHAKE-256 expansion uses
   separate `input-r` and `output-s` purpose domains, bound to inventory,
   model/body/weight, stage, attempt, shape, quantization and ring metadata.
2. Offline Prep computes `c = W r − s mod 2^k`; inference gets **only c and
   public identifiers/metadata**, not the root, `r`, or `s`.
3. Online client sends `u = x − r mod 2^k` to inference. Inference computes
   `v = W u + c = W x − s mod 2^k` and returns it to client.
4. Client recovers `v + s = W x mod 2^k`, signed-decodes within the existing
   declared output bound, then applies existing scales/bias/nonlinear handling.

Placement changes none of these equations or arithmetic contracts. This is an
exact modular identity, **not** a new security proof. The reference tests cover
u16/u24/u32 wraparound and signed recovery with production mask expansion and
correction/request serialization.

One root per stage/inventory produces multiple row masks. It is not one root
per row; each derived row correlation still has a one-use attempt identifier.
Roots, contexts and used rows must never be replayed. Reservation advances
monotonically; unused reserved rows burn on termination. The 39-input/32-output
response requires **39 + 31 = 70 fresh rows per stage**, since the final emitted
token needs no further decode step. Archived baseline: 96 stages / 6,720 stage
rows. Attention variant: 48 stages / 3,360 stage rows. Both consumed 70 rows,
burned zero, reused zero. Warmup used a separate fresh inventory.

### Boundary exactly

| Placement | Client ↔ Prep | Prep ↔ inference | Client ↔ inference | Trust requirement |
|---|---|---|---|---|
| Ordinary remote Prep | WAN | WAN | WAN | Prep/inference must not collude |
| User-owned separate appliance | Trusted LAN | WAN | WAN | User trusts appliance OS, storage and root handling |
| Prep on client host, existing transport retained | Trusted loopback | WAN | WAN | User trusts client host and local Prep process |
| Client-host IPC counterfactual | Logical local IPC, no network body | WAN | WAN | Same local trust; IPC replacement unimplemented |

External operator non-collusion dependency can disappear **only when Prep is
genuinely user-controlled and its secrets remain inside the user's trusted
boundary**. A provider-managed appliance or compromised local OS does not
establish that property. A Prep compromise plus captured online masked inputs
can recover activations. The same privacy goal remains: remote inference does
not receive plaintext prompt/token IDs, activation masks, roots, or output
masks. Root compromise, replay/rollback resistance, adversarial correctness and
metadata leakage remain obligations of the existing protocol.

The baseline carries 96 × 32 = **3,072 root bytes** client→Prep per inventory;
attention carries 48 × 32 = **1,536**. Ordinary remote Prep receives them over
WAN. Local Prep receives them over LAN/loopback/IPC; **zero root bytes need WAN
or inference**. Actual messages include identifiers/authorization beyond those
root bytes. Derived opaque row IDs may cross WAN; they are not root transfer.

This is local placement of the preparation role, not a customer gateway
decrypting provider results or running the whole decoder. Existing client
nonlinear work stays client-owned. The attention row below is a separately
archived client-attention workload, not a claimed improvement caused by local
Prep.

## Exact source and workload locks

- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`; W8/A8.
- Body fingerprint:
  `5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974`.
- Source lock:
  `880f80ec61274d3c80e2a0c1336394b9e955d53ff98436a426a680a460a592e7`.
- Warm prompt digest:
  `c32aa8d3d5c293a2a3f3b01c6de6984ea2a6525855ef1578b63efcb95aa040d3`.
- Cold prompt digest:
  `80f8e4ec58669b73af23235a8aa0c7ec2d4f7bbb9e6d8f8efe3c22629aebaaa4`.

Each placement comparison reuses the **identical directed sample**, including
its prompt digest, model, row count, inventory policy and body encoding. The
warm baseline and attention samples share their comparison key; the cold pair
also share theirs. Warm and cold are distinct prompt-digest cohorts and must
not be represented as a same-prompt latency comparison.

Inputs:

- `prepared-stage-attribution-qwen25-2026-09-29.json` provides compiler-bound
  semantic-role totals: 178,967,744 attributed stage bytes plus 2,814 covered
  authorization bodies = 178,970,558 warm baseline bytes.
- `incremental-network-qwen25-2026-10-01.json` provides exact-source cohort keys,
  baseline/attention totals, CPU samples and body-placement resources. Its
  summary alone does **not** retain directed edges.
- The existing `qwen-incremental-32-warm.json` and
  `qwen-incremental-32-cold.json` raw reports in the pre-approved temporary
  archive contain reconciled directed stage/body ledgers for both variants.
  Their original SHA-256 hashes and compact extractions are retained in the
  new screen JSON. Default reproduction no longer needs the temporary files.

The old stage summary has no prompt digest; its role totals are a byte
cross-check, not independent evidence of prompt identity. The raw incremental
records supply the matched comparison keys. Source document hashes and the
canonical compact-ledger hash are enforced by the probe. Modified sources,
missing/negative/duplicate edges, nonzero reuse, changed keys, or silent
directed-byte redistribution fail closed.

## Directed application-body ledger

All numbers below are **bytes**, one counted sending-side body per directed
edge. These are not full wire measurements. Warm includes a new request-sized
inventory and online execution; it excludes the already-delivered bundle.

| Directed body / phase | Baseline warm | Attention warm | Local-Prep locality |
|---|---:|---:|---|
| Client → Prep / offline, including authorization | 12,658 | 6,390 | LAN or loopback |
| Prep → client / offline, including acknowledgement | 8,620 | 4,272 | LAN or loopback |
| Prep → inference / offline correction upload | 65,404,256 | 55,066,276 | **WAN** |
| Client → inference / online masked inputs | 47,703,296 | 37,936,396 | **WAN** |
| Inference → client / online masked outputs | 65,841,728 | 55,284,652 | **WAN** |
| Inference → client / warm bundle | 0 | 0 | WAN when delivered |
| **Warm total** | **178,970,558** | **148,297,986** | |
| Inference → client / cold bundle, separately added | 145,977,006 | 190,217,984 | **WAN** |
| **Cold covered total** | **324,947,564** | **338,515,970** | |

Per-role stage attribution (warm; excludes the authorization remainder):

| Role, all 24 layers | Client→Prep | Prep→client | Prep→inference | Client→inference | Inference→client | Total |
|---|---:|---:|---:|---:|---:|---:|
| QKV projection | 2,472 | 2,198 | 5,814,134 | 4,884,194 | 5,924,402 | 16,627,400 |
| Attention output | 2,472 | 2,150 | 4,523,846 | 4,882,706 | 4,632,674 | 14,043,848 |
| MLP gate/up | 2,472 | 2,150 | 49,037,126 | 4,882,706 | 49,145,954 | 103,070,408 |
| MLP down | 2,472 | 2,078 | 6,029,150 | 33,053,690 | 6,138,698 | 45,226,088 |
| **Baseline stage sum** | **9,888** | **8,576** | **65,404,256** | **47,703,296** | **65,841,728** | **178,967,744** |

Attention's two MLP rows are exactly the same archived directed bodies as the
baseline MLP rows. Its stage total is 148,296,496; covered authorization
remainder is 1,490. Cold bundle delivery adds no semantic-stage attribution.
These totals reconcile all five protocol stage counters independently.

## LAN vs WAN vs all-link: same 39+32 workload

### Warm, including recurring offline preparation

| Variant / Prep placement | LAN | WAN | Loopback | All network links | Logical IPC excluded from network |
|---|---:|---:|---:|---:|---:|
| Baseline / remote | 0 | 178,970,558 | 0 | 178,970,558 | 0 |
| Baseline / local appliance | 21,278 | 178,949,280 | 0 | 178,970,558 | 0 |
| Baseline / client-host loopback | 0 | 178,949,280 | 21,278 | 178,970,558 | 0 |
| Baseline / client-host IPC counterfactual | 0 | 178,949,280 | 0 | 178,949,280 | 21,278 |
| Attention / remote | 0 | 148,297,986 | 0 | 148,297,986 | 0 |
| Attention / local appliance | 10,662 | 148,287,324 | 0 | 148,297,986 | 0 |
| Attention / client-host loopback | 0 | 148,287,324 | 10,662 | 148,297,986 | 0 |
| Attention / client-host IPC counterfactual | 0 | 148,287,324 | 0 | 148,287,324 | 10,662 |

**Logical all-link totals remain 178,970,558 / 148,297,986 in every row.**
WAN reduction is only about **0.01189% baseline / 0.00719% attention**.
Online-only WAN remains **113,545,024 / 93,221,048** for every placement.
Charging correction upload to an earlier offline window does not remove it
from recurring bytes per inventory/response.

### Cold covered bodies, excluding unmeasured checkpoint distribution

| Variant / Prep placement | LAN | WAN | Loopback | All network links | Logical IPC excluded |
|---|---:|---:|---:|---:|---:|
| Baseline / remote | 0 | 324,947,564 | 0 | 324,947,564 | 0 |
| Baseline / local appliance | 21,278 | 324,926,286 | 0 | 324,947,564 | 0 |
| Baseline / client-host loopback | 0 | 324,926,286 | 21,278 | 324,947,564 | 0 |
| Baseline / IPC counterfactual | 0 | 324,926,286 | 0 | 324,926,286 | 21,278 |
| Attention / remote | 0 | 338,515,970 | 0 | 338,515,970 | 0 |
| Attention / local appliance | 10,662 | 338,505,308 | 0 | 338,515,970 | 0 |
| Attention / client-host loopback | 0 | 338,505,308 | 10,662 | 338,515,970 | 0 |
| Attention / IPC counterfactual | 0 | 338,505,308 | 0 | 338,505,308 | 10,662 |

The 190,217,984-byte attention bundle exceeds baseline by 44,240,978 bytes;
the archived client-attention placement owns 44,040,192 int8 matrix bytes plus
196,608 scale bytes and bundle metadata. That is real extra client storage and
client linear work, distinct from placing Prep locally. Attention warm
all-link reduction (~17.14%) comes from its client-owned attention stages;
its larger cold bundle can make cold covered bodies worse. Local Prep alone
does not create that attention benefit.

## Body lower bound and resource admission

Public geometry, checked against archived declared MACs: 24 layers; hidden
896; QKV output 1,152; gate/up output 9,728; down input 4,864. A fresh row
requires matrix-vector products for each remote stage, regardless of where
Prep runs. A warm weight cache reuses **weights**, not mask rows or `W r`.

| Requirement / 70-row inventory | Baseline | Attention variant |
|---|---:|---:|
| Active Prep int8 matrix payload floor | 357,826,560 | 313,786,368 |
| Corresponding fp32 scale payload | 1,216,512 | 1,019,904 |
| Prep linear MACs per inventory | 25,047,859,200 | 21,965,045,760 |
| Client expanded uint32 input/output masks | 135,905,280 | 110,100,480 |
| Remote uint32 correction-array payload | 85,155,840 | 71,393,280 |
| Conservative u16 correction-tensor body floor | 42,577,920 | 35,696,640 |
| Measured actual correction body | 65,404,256 | 55,066,276 |

The minimum correction floor uses the existing protocol's smallest supported
16-bit ring, **not** an assertion that the locked stages can switch to u16.
The raw reports retain body attribution, not exact per-stage ring profiles;
actual mixed-ring widths are unknown here. The floor excludes headers and
identifiers. Even that optimistic correction floor alone exceeds each
variant's 10× warm all-link budget (17,897,055 / 14,829,798 bytes). Existing
online body plus that floor is at least **156,122,944 / 128,917,688 bytes**
without any further protocol change. Moving Prep cannot approach 10×.

Storage and RAM are not interchangeable. Full public body int8 snapshot alone
contains **357,826,560 bytes**, plus **1,216,512** body scales, even when only
MLP matrices are active for Prep. Do not assume attention placement deletes
the rest of Prep's model. Checkpoint, embedding/head/nonlinear tensors,
quantization/import buffers, native compiled copies/caches, Python objects and
allocator overhead add storage or memory. Checkpoint distribution and total
persistent snapshot/cache disk sizes are **unmeasured**, not zero. Initial
public-model distribution can be amortized across uses of an unchanged pinned
model, but neither its bytes nor its transfer time was measured.

Existing client caches expanded `r/s` arrays. Local appliance placement does
not automatically move those arrays away from client or make it seed-only.
Same-host Prep adds preparation weights and transient compute buffers to the
client machine. A spare 70-row inventory doubles the mask-array payload and
remote correction storage; queues of N such inventories scale those payloads
by N. Prep need not retain all correction arrays permanently, but current
matrix/mask expansion, int64 intermediates and serialization have nonzero
peak RAM. Server correction-array counts are array payloads, not measured
resident peaks or an assertion about the full storage representation.

Archived single-host CPU (seconds; cumulative cold totals include startup):

| Sample / role | Baseline | Attention |
|---|---:|---:|
| Warm measured window / Prep | 6.414704384 | 5.551877632 |
| Warm measured window / client | 7.192128 | 8.501883000000007 |
| Warm measured window / inference | 10.235867392 | 7.672685568000002 |
| Cold startup / Prep | 7.989242112 | 8.072429056 |
| Through first cold response / Prep | 14.449014656000001 | 13.886022912 |
| Through first cold response / all roles | 41.604709848 | 40.768367848000004 |

These establish that local Prep is substantive CPU work; they do not predict
home-appliance performance or representative throughput. No compute-cap,
peak-RAM or full-disk admission was established. **Resource admission remains
unknown.** A device below a required array/storage floor is ruled out; passing
the floors is not sufficient admission. Admission needs measured complete
model/cache footprint, peak client+Prep RAM for same-host placement, sustained
fresh-row generation/upload capacity and response concurrency on the proposed
hardware. Heavy online inference remains remote in both variants.

## Availability and inventory lifecycle

- **Before online:** generate fresh stage roots/masks, compute and upload all
  corrections, match inference acknowledgements, then seal the inference
  inventory as `ready`. Existing `preparation_server.py` checks acknowledgement
  attempt/stage/byte count; `client.py` accepts matching Prep acks and commits
  `/ready` before returning the inventory. No optimistic use of unacknowledged
  material is licensed by relocation.
- **During an active response:** no preparation/refill. Existing activity
  guards reject `_begin_preparation()` while inference is online; background
  refill starts only when active response count is zero. Always-on means idle
  preparation opportunity, not concurrent online refill.
- **Prep unavailable after sealing:** a response can proceed conditionally if
  client still holds enough reserved masks and remote inference retains the
  same live inventory. The one sample is not an appliance-disconnection test.
  Inference expiry/restart or client state loss invalidates that assumption.
- **LAN outage with a separate appliance:** already-sealed rows may still be
  usable, but fresh refill is unavailable. Remote inference still requires WAN.
  Same-host placement avoids this LAN link but shares client uptime and RAM.
- **Inference/WAN unavailable:** no correction acknowledgement/seal and no
  remote online response. Local Prep is not an offline inference fallback.
- **Exhaustion, expiry, cancellation or partial failure:** wait for a fresh
  inventory before another online response; consumed or burned rows never
  return to the reservoir. Snapshot rollback, durable one-use state and
  multi-session/concurrency availability have not been measured here.

Byte ledger omissions remain explicit: HTTP/TLS/WebSocket framing, retries,
control/seal/status bodies, Prep→inference authorization forwarding and
inference→Prep correction acknowledgements are not all covered by the archived
client-ledger counters. Those last two cross WAN in the local-placement model.
Their absence prevents a full-wire claim and must not be treated as zero.

## Reproduction and checks

No model download, model execution or provider startup required:

```bash
rtk proxy uv run --no-sync python scripts/probe_preparation_placement.py
rtk uv run --no-sync pytest -q tests/test_preparation_placement.py
rtk uv run --no-sync ruff check scripts/probe_preparation_placement.py tests/test_preparation_placement.py
```

Optional original-archive verification, while those archived files exist:

```bash
rtk proxy uv run --no-sync python scripts/probe_preparation_placement.py \
  --archive-dir /var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode
```

`--section placement_table`, `--section resource_floors`,
`--section archived_ledgers` or `--section retained_directed_ledger_sha256`
prints a selected result. Default output reproduces the saved JSON structurally
exactly, including locked cold CPU evidence. Tests verify locality sum
conservation, IPC exclusion charged separately, correction uplink retained,
source/cohort locking, directed-mutation rejection, independent signed-ring
recovery, production serialization without roots to inference, purpose/attempt
mask separation, and burned-row/no-reuse inventory behavior.

## Recommendation

Retain as a **user-controlled trust-placement option for future measurement**,
conditional on local-device resource and inventory-lifecycle admission. It can
replace an external non-collusion dependency with explicit user-device trust
and may alter latency/local operational control. Archived bodies show
negligible WAN-byte improvement, no appliance all-link reduction, substantial
recurring Prep CPU and large public-weight/mask storage. Do not promote it as
a network-efficiency breakthrough or an admitted runtime topology.
