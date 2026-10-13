# Optional attested linear executor: bootstrap and byte budget

## Recommendation and evidence boundary

**Explore a separately opt-in full CPU-enclave linear executor, not a replacement
for ordinary private inference.** Source-hashed geometry supports roughly **39.49%
less first-pair application-body traffic** and **52.87% less recurring arithmetic
traffic** with fixed public-bound bit packing. This is a budget projection, not a
measured deployment, throughput result, security endorsement or TEE proof.
35% is plausible only with priced setup; 50% requires amortization and tight
additional-cost control. Do not promote either claim to production.

Existing `python/pllm/deployment/role_placement.py` and
`schemas/role-deployment.schema.json` define read-only placement/pinned policy
declarations. `PlacementAssessment` defaults attestation verification and runtime
admission to false. Declared technologies are AMD SEV-SNP, Intel TDX, Arm CCA and
NVIDIA confidential GPU; none is evidence of an available attested runtime.
Nitro, discussed below, is not an existing schema technology. No contracts,
production code or policies change in this investigation.

Ordinary prepared inference retains trusted Preparation, one-use masks,
erasure and non-collusion assumptions; see `network-algebra.md`,
`correlation-compiler-reference.md`, and `network-io-10x.md`. Here the executor
learns quantized activations and relies on hardware isolation, measured software,
attestation roots, firmware/TCB policies and a correct verifier. Client scales,
nonlinear work, KV, tokenization and public boundary work remain client-owned.
Private scales do **not** make exposed quantized activations harmless. Host still
observes schedule, lengths, timing, termination and access-pattern leakage unless
separately mitigated. Side channels, rollback, malicious guest administration,
debug paths, denial of service and vendor compromise need explicit threat review.

## Two mechanisms: costs cannot disappear through placement labels

| Mechanism | Trusted execution and accounting | Recommendation |
| --- | --- | --- |
| Full linear execution | Enclave receives end-to-end encrypted signed A8 values, owns immutable source-locked i8 weights, computes exact `W*x` integers, checks public bounds, packs before AEAD. No `W*r-s` correction or wide input mask. | Preferred research candidate. CPU enclave includes all linear execution; confidential GPU is a different platform profile. |
| Local enclave correlations, external masked evaluator | Enclave expands fresh `r,s`, computes `W*r-s`, sends correction and wide masked `x-r` to host; host evaluates masked stage, enclave restores/bounds/encrypts result. | Not the same saving. Adds `W*r`, mask expansion, internal transport/copies, unmasking and host correctness problem. |

Public geometry totals **56,615,845,888 body MACs per cold/repeat pair**.
Correlation mechanism performs approximately another such `W*r` workload as well
as host masked evaluation, rather than making Preparation computation free.
Count enclave→host masked inputs, corrections, host→enclave outputs, socket/ring
buffers, host/device transfers and copies separately from external client links.
The historical 139.06 MB correction class is not an actual measured IPC ledger;
renaming this edge IPC does not prove bytes eliminated. If enclave uses external
weights, authenticate every page/tile and pin its version before use.
An AEAD tag on a host-supplied result authenticates the enclave's message, **not
host arithmetic correctness**. Recompute, a reviewed verifiable scheme or an
explicit weaker honest-evaluator assumption is required; no such scheme is proved
here. Correlation route cannot inherit full-executor correctness merely by
encrypting replies. Untrusted GPU receiving clear A8 is out of scope.

## Bootstrap and lifecycle design (not implemented)

1. **Pin source and software.** Client pins model revision, checkpoint digest,
   source-lock/body/quantizer/numeric-plan commitments, exact weight tile manifest,
   public bounds and execution schedule. Pin provider identity, measurements,
   verifier roots, minimum TCB/security versions, no-debug configuration, expiry,
   revocation policy and permitted migration. Measured code must verify imported
   artifacts: a launch measurement alone does not authenticate subsequently
   loaded weights. Host-writable weight caches require hashes/Merkle proofs against
   signed, client-pinned manifest and immutable verified enclave copies; hashing
   then reading mutable host memory creates a TOCTOU gap. Account reading,
   verification, staging, duplicate files and page faults.
2. **Fresh direct attestation preferred.** Client challenges executor with fresh
   unpredictable nonce. Verify quote signature chain, freshness, current
   endorsements/revocation, measurement, TCB, debug status and report-data binding
   to enclave-generated ephemeral public key, source/plan commitments and provider
   epoch. Bind transcript and identity to key derivation; mutually confirm keys.
   Hardware evidence and application-source evidence must both be checked.
   Direct client verification avoids giving Preparation power to approve arbitrary
   executors or possess session plaintext keys. Endorsement/verifier services still
   add trust, CPU and traffic.
3. **Optional bootstrap authority.** Preparation may publish signed policy or
   artifact commitments or attest/key-release once per source/measurement/provider
   epoch. Client must pin that authority, bound its power and charge policy, quote,
   key-release, revocation and distribution edges. Delegating verification adds
   authority compromise risk; require transcript-bound verifier receipts and
   explicit client consent. Preparation need not see traffic secrets. It must not
   become an uncharged per-request authorization or ciphertext relay. Nitro's
   untrusted parent byte relay is different: charge IPC/copies even if no separate
   Preparation authority participates.
4. **Persistent keys, bounded sessions.** Retain only protected epoch master/key
   handles, not reusable masks. Derive independent directional traffic keys per
   unpredictable session and renewal generation with transcript-bound HKDF.
   Bind source/measurement/provider epoch, session, phase, stage, counter, public
   row/length/width and framing version in authenticated header/AAD. Validate
   public input dimensions and legal schedule; no arbitrary chosen weight/stage
   execution. Scale/bias placement and integer accumulator/overflow behavior must
   match original numeric plan; client performs original dequantization/nonlinear
   steps. Authenticated integers alone do not prove complete decoder parity.
5. **Rollback and retries.** Strict monotonic counters per directional key; bounded
   out-of-order window only if explicitly designed. Never encrypt different bodies
   under same nonce. This probe uses 4-byte nonce prefix + 8-byte counter under
   distinct directional/session keys; prefix collision is not protection against
   restored state with the same key. Durable client checkpoint must not restore
   old traffic counters. Enclave reset, uncertain counter durability, restored VM,
   cloned session or failover must burn old session, establish new unpredictable
   generation and keys, and obtain fresh attestation when epoch changes. Sealing
   alone does not establish freshness; require an external monotonic witness or
   fail-closed fresh key establishment. Account witness traffic if used. Identical
   retransmission may use cached ciphertext under defined idempotency; otherwise
   consume new counter. No client state advances on missing/invalid authenticated
   output. Bound failure retries and account every attempt and partial body.
6. **Renew and revoke.** Re-attest at bounded lease/TCB-policy expiry, source or
   code update, provider change, hardware reset and migration. Offline operation
   needs maximum stale-collateral interval and explicit unavailable-verifier
   behavior. Revocation stops new admissions, cancels outstanding sessions and
   burns counters/keys; no fallback to clear inputs or falsely labelled ordinary
   private inference. Fresh sessions derive fresh keys; a new epoch never silently
   reuses source commitments. Price periodic revocation checks and renewals rather
   than claiming permanent once-only attestation.
7. **Close.** Authenticated bounded shutdown/ack, invalidated key handles, burn
   queued stages and retries, zeroize private buffers/key material, disable logs,
   dumps and unsafe swap. Crash relies on epoch invalidation and fresh handshake,
   not successful close acknowledgement. Record failure/close traffic and audit
   metadata without recording activations. Python `close()` in this probe drops
   references only: no secure zeroization or durable lifecycle is implemented.

## Reproducible geometry and AEAD budget

`scripts/probe_attested_linear_budget.py` reads existing public evidence only:

- `research-all-link-frontier.json`: historical **493,575,247** body bytes:
  corrections 139,060,689; outputs 139,123,150; inputs 90,443,518; common public
  bundle 124,905,091; controls/seeds 42,799. All directions counted once.
- `network-algebra-qwen25-2026-10-06.json`: same body fingerprint, 96 source-hashed
  stage matrices. Uses its recorded public stage output bound
  `(odd_radix.modulus-1)/2`, not private activation-dependent widths. The historical
  geometry probe's 39-token execution is **not** reused as this pair's row count.
- Maintained `research-prepared-study-qwen25-confirm.json`: 150+8 cold and exact
  repeat requests. Cold prefill 150 rows, seven decode rows per request; repeat
  prefix hit skips prefill. Final layer output projection and two MLP groups use
  one terminal prefill row; qkv remains full for KV. **93×164 + 3×15 = 15,297**
  stage rows, independently matching all three measured material-claim ledgers.
  Compiler pruning source/test hashes and source-lock identity are recorded.

The original symmetric A8 quantizer permits **−127 through 127**, not −128.
`transformer_engine.py` defines recorded bound as `127 * max_row_L1(W)`; executor
must reject −128 under this numeric plan. A byte container does not authorize a
wider activation range. Conservative fan-in variant deliberately uses 128 for
both operands. Synthetic transport test separately exercises full i8 extremes
with its own 128-based public bound, not the original model quantizer.

No actual weights are loaded or re-quantized. Weight hashes and bounds are inherited
from frozen geometry, not independently regenerated. A future executor must
verify them against actual imported weights, scales and bias/numeric plan.

Frame estimate uses 16-row prefill chunks and individual decode rows:
**2,277 frames each direction**, 70-byte authenticated header, explicit 12-byte
nonce, 16-byte AES-GCM tag, total **98 bytes overhead/frame**. Header contains
version, 16-byte session ID, 32-byte combined source/measurement/provider epoch
commitment, phase, stage, 64-bit counter, rows, plaintext length and public width.
Epoch manifest itself remains separately charged bootstrap material. Fixed packed
output widths are 23/24/26 bits for recorded stage bounds; pack signed exact
integers **before** AEAD with canonical zero padding. No ciphertext compression.

| Pair payload | Bytes |
| --- | ---: |
| Signed A8 inputs | 28,732,928 |
| Fixed recorded-public-bound output bits | 144,546,192 |
| Both-direction AEAD framing | 446,292 |
| Conservatively retained historical controls/seeds | 42,799 |
| Candidate recurring pair | **173,768,211** |
| Common public delivery, once per horizon | **124,905,091** |

Historical recurring pair is **368,670,156**. Retain output traffic; simply removing
corrections and assuming unchanged historical replies gives an overoptimistic
ceiling. Public-bound unmasked packing differs from historical masked residue
encoding. Conservative fan-in bound `128*128*in_features` needs 151,774,032 reply
bytes; raw signed int32 needs 192,642,048 bytes. Header estimate excludes HTTP/TLS,
ACKs, retransmissions and auxiliary lifecycle packets on **both** deployments.

### Fair shared-delivery horizons

For `h` fresh-prefix/exact-repeat pairs, baseline **also** amortizes shared bundle:
`B(h) = P + h*R`, candidate `C(h) = P + h*R' + E`.
`E` is extra candidate setup over equally charged baseline setup, not a measured
64 KiB handshake. For recurring extra costs `V`, use `R'+V`; add every renewal,
failed attempt, storage relay and distribution difference at its actual horizon.
Target `t` permits `E <= h*((1-t)*R-R') - t*P`.
Negative allowance means target impossible at that horizon even before setup.

| Pairs | Public-bound packed saving before extra costs | Extra setup allowance for 35% | Extra setup allowance for 50% |
| ---: | ---: | ---: | ---: |
| 1 | 39.49% | 22.15 MB | −51.89 MB |
| 2 | 45.21% | 88.02 MB | −41.32 MB |
| 4 | 48.74% | 219.75 MB | −20.19 MB |
| 8 | 50.72% | 483.22 MB | 22.08 MB |
| 16 | 51.77% | 1,010.16 MB | 106.62 MB |

Decimal MB. At first pair, 20% allows 96.19 MB extra setup. With zero extra setup,
public-bound packing reaches 20/35/50% at **1/1/6 pairs**; a hypothetical 65,536
extra bytes gives same rounded horizons, not evidence of actual attestation size.
Charging one additional 357,826,560-byte raw body-weight transfer reaches targets
at **4/7/40 pairs**. Charging one additional 988,097,824-byte original checkpoint
reaches **9/16/100 pairs**. These are deliberately extra-cost sensitivities, not
claims that all deployments need duplicate source delivery.

Conservative fan-in packing reaches zero-extra 50% only at **19 pairs** and tends
to 50.91%; almost no recurring-cost headroom remains. Plain int32 reaches 20/35%
at **1/3 pairs**, tends to **39.82%**, and can never reach 50%. At large horizons,
public-bound packing's 50% recurring headroom is only **10,566,867 bytes/pair**;
that is not permission to exclude local transport or renewal costs.

### Source and resource owners outside serving ledger

Separate cold-distribution payload scenarios, inherited from frozen publisher
screen, are recorded in JSON: two original provider checkpoints 1,976,195,648
bytes versus one candidate checkpoint 988,097,824; role-artifact baseline
855,114,752 versus a conservatively retained single Inference artifact 496,071,680.
These raw-payload projections exclude manifests, signatures, tokenizer/index,
publication/quantization CPU, publisher source retention and storage. They cannot
be subtracted from a serving ledger that never charged those downloads. Compare
`S_baseline + P + hR` with `S_candidate + P + hR' + lifecycle(h)` once actual
distribution owners are known. Shared URL does not make separate transfers free.

Executor body weights alone occupy **357,826,560 bytes**, before row scales,
runtime, buffers and verified staging. CPU-enclave paging must authenticate host
pages and price memory bandwidth, rereads, page cache and enclave/host copies.
Client still owns tied public vocabulary matrix, scales/nonlinears, tokenizer,
KV/history and response buffers. **256 MiB client target remains plausible, not
validated**: historical paged-client peak does not transfer automatically. Measure
fresh lifetime peak including handshake/verifier libraries, code/crypto buffers,
concurrency and public delivery. Enclave/guest RSS, host RSS, GPU device memory,
page/file cache, storage and shared mappings are separate owners, not client
memory savings. Smaller network does not imply lower CPU, latency or cost.

## Achievable platform classes from primary documentation

Reviewed public vendor docs 2026-10-13; no account, provisioning or expenditure:

- [AWS Nitro Enclaves](https://docs.aws.amazon.com/enclaves/latest/user/nitro-enclave.html):
  isolated constrained VM with dedicated parent vCPU/memory, Linux enclave,
  no persistent storage or external networking, parent-only local sockets and
  attestation. Supports CPU linear executor with parent encrypted-byte relay;
  parent IPC/artifact load and lifecycle costs remain. Supported GPU parent
  instance **does not establish enclave GPU execution**. Nitro is not presently
  represented in this project's placement technology enum.
- [Azure confidential VMs](https://learn.microsoft.com/en-us/azure/confidential-computing/confidential-vm-overview):
  AMD SEV-SNP/Intel TDX guest attestation and hardware isolation from hypervisor
  and host management; confidential CPU VM class is feasible substrate. Guest
  OS/admin is inside trusted boundary, unlike treating provider-controlled guest
  as hostile. Pin measured immutable guest and executor, restrict administration,
  verify imported artifacts and key release. Provider platform boot attestation
  alone is not executor-specific source verification. Storage, swap and VMGS have
  independent configuration/cost; check current support before choosing a SKU.
- [NVIDIA trusted computing docs](https://docs.nvidia.com/confidential-computing/index.html):
  deployment guide explicitly covers TDX/SNP plus supported NVIDIA GPUs, separate
  GPU attestation and GPU/VBIOS/CUDA/CC-mode compatibility. CPU+GPU confidential
  profile needs combined measured software/device chain and protected transfers;
  GPU quote alone does not measure application kernels or weight correctness.
  New drivers, firmware, device memory and encrypted transfer resources change
  platform/scope; no claim that ordinary host GPU becomes safe through CPU quote.

These identify available architectural classes, not tested products, price quotes,
capacity commitments or security guarantees. No hardware quote is synthesized.

## Tiny validation and replay

```sh
target/frontier-sdk-env/bin/python scripts/probe_attested_linear_budget.py \
  --output docs/evidence/research-attested-linear-bootstrap.json
target/frontier-sdk-env/bin/python -m pytest -q tests/test_attested_linear_budget_probe.py
```

Probe uses installed `cryptography` 46.0.7, no wheel rebuild/install, checkpoint
download or full response. Thirty-two public synthetic 3×11 inputs and 7×11 i8
matrices check **672 exact output integers**, public L1 bounds, fixed bit packing,
AES-256-GCM round trips, tamper and replay rejection, charging **8,866 bytes**.
Includes extreme −128/127/zero vectors. Keys are public deterministic test keys,
not fresh secret provisioning; directional HKDF is a software control only.
Synthetic parser cap is intentionally tiny, not an executable model-frame codec
or a measured implementation of projected large frames. Python arithmetic avoids
overflow; native parity still needs measurement.

Focused suite: **34 passed**. Tests cover canonical signed packing, width/length
bounds, authenticated phase/stage/rows/width, wrong session/epoch/key/direction,
tamper without counter advance, replay, truncation, exhaustion/close, historical
row/body/weight conservation, changed evidence rejection and independent horizon
oracle. Source SHA-256s and software dependency version accompany JSON.

Missing promotion evidence: real nonce-bound quote verification and revocation;
source import/weight-page integrity; durable renewal/rollback/failure/close tests;
actual every-edge bootstrap/IPC/device/wire accounting; exact complete logits/KV
and integer-kernel parity; all-role fresh peak RSS/CPU/device/cache ownership;
latency/throughput and bounded failure behavior; side-channel/security review and
hardware availability. No complete runtime, TEE proof or production admission
exists here.
