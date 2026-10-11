# Finite encrypted residency probe

`scripts/probe_encrypted_residency.py` advances the existing independent CKKS
linear foundation to a complete, bounded state boundary. The evaluator retains
encrypted state for two transitions and emits only a two-coordinate public head:

```
h_next = h + (W h) ** 2
y = H h_final
```

This is a synthetic polynomial recurrence, not a replacement trained architecture,
attention layer, checkpoint decoder, token selector, KV cache or private feedback
loop. Its utility is executable composition and amortization, not model quality.
SEAL has no bootstrapping. Depth exhaustion rejects; no intermediate decrypt or
secret-key refresh substitutes for missing bootstrap.

## Matched control and candidate

Both run eight independent width-eight vectors, the same public matrices, two
transitions, the same degree/modulus/scale, separate client-secret/provider-public
contexts and one fresh keypair per process. The control issues eight independent
randomized ciphertext queries serially with reused evaluation keys. The candidate
issues one ciphertext containing eight coordinate-major interleaved lanes.
Every packed lane belongs to **one client and one key context**. This does not
establish cross-client multi-key batching, isolation between separate clients or
collaborative decryption.

Coordinate `c`, lane `q` occupies slot `c * lanes + q`. Complete periods repeat
across all slots. A single Galois key rotating by `lanes` slots implements diagonal
matrix multiplication without cross-lane wrap contamination. This differs from
naively concatenating independent vectors, which leaks arithmetic across blocks.
The control uses a one-slot key; the candidate uses an eight-slot key. Key count,
serialized padding and admitted memory match, but key values necessarily differ.

Every transition consumes one plaintext-matrix rescale and one ciphertext-square
rescale/relinearization. Residual addition explicitly aligns modulus levels. The
final head consumes another level: depth five total, degree 16,384, coefficient
bits `[50,40,40,40,40,40,50]`, scale `2^40`, mandatory SEAL TC128 admission.
One to three transitions are admitted; arbitrary recurrence is not.

Final head pads unreturned coordinates with public zeros. The client decrypts the
whole final ciphertext and all repeated output/padding slots are checked against
an independent `math.fsum` dense recurrence at absolute tolerance `0.0002`.
Approximate CKKS zero padding is not circuit privacy. No claim of exact integer
parity, hidden public weights, malicious-query security or authenticated transport
is made. Head extraction does not shrink a SEAL ciphertext; batching amortizes
its fixed capacity across independent inputs instead. This targets throughput of
independent concurrent queries; it cannot batch causally dependent decode tokens
from one stream or remove latency waiting for a batch to fill.

## Cost and ownership

Each sample charges parameter/public/Galois/relinearization delivery once, every
query/reply frame including padding and 81-byte headers, public matrix/diagonal
payloads, secret polynomial, public evaluation-key payloads, two distinct contexts
and sixteen live-ciphertext admission slots. Existing native admission accounts
for serialized delivery overlap, contexts/RNS scratch, ciphertexts and pool slack.
This estimate is not measured per-party residency or an OOM proof. No one-use
garbling/dealer material is involved; HE evaluation keys legitimately reuse within
the declared session. Both roles retire on success/failure through existing owners.

Serial/packed workers run in fresh processes under the shared exclusive frontier
lock, 512 MiB host-headroom admission, memory/swap watchdog and fixed thread caps.
Report separates setup wall/process CPU, online wall/process CPU, client online
CPU, evaluator CPU, whole-process peak RSS and frame totals. Setup-inclusive frame
ratio is mandatory: an online-only factor is not an all-link or whole-response
result. Local frame serialization disk work enters timing and unpadded file
payload writes/reads are charged; physical disk I/O and transport remain unknown.
Oracle validation time is separated from online timings. Seven whole-response
serving metrics stay null.
Probe, shared helper and actually imported runtime/build/memory/shutdown source
paths and hashes freeze before workers. Each worker checks this identity before
crypto and after execution; the parent verifies it before workers and after the
cohort. Code source identity stays separate from intentionally different public
rotation steps, which are recorded per control/candidate. Native implementation
identity and installed build also bind reports. Whole-process `ru_maxrss` is
normalized to bytes (Darwin bytes; non-Darwin KiB). No inputs, decoded outputs,
token IDs, randomness, keys or their
hashes enter persisted evidence.

## Stage-one checks and deferred serial execution

```sh
/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python -m pytest --noconftest -o pythonpath= -q tests/test_encrypted_residency_probe.py
/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python scripts/probe_encrypted_residency.py --prepare
```

Clear-only tests check independent dense recurrence, all periodic output slots,
zero head padding, zero-state lane isolation, rotations and invalid geometry.
`--prepare` calls native public admission only; it generates no crypto material.
Native admission estimates 189,255,680 bytes combined allocation. Clear-only tests
use `--noconftest` to avoid shared model/gateway fixtures and source-tree native
imports; they need no wheel build.

The executable public padded-frame gate predicts 23,925,248 setup bytes in each
mode, 14,746,896 serial online bytes versus 1,843,362 packed online bytes. Thus
online ratio is eight, but setup-inclusive ratio is only approximately 1.50 for
this eight-query lifecycle. This is a public geometry calculation, not an executed
crypto or physical-wire measurement. Serial execution must match these byte counts
exactly before evidence is accepted. Changing key lifetime or ignoring setup would
change the comparison and is not silently permitted.

After coordinator authorizes serial crypto/timing:

```sh
/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python scripts/probe_encrypted_residency.py --processes 3 --output docs/evidence/research-encrypted-residency-cost.json
```

No native changes or wheel rebuild required. Cohort uses six fresh workers and
512 MiB host-headroom admission. No 100x network/client-CPU gain is asserted.

## Measured serial cohort

`docs/evidence/research-encrypted-residency-cost.json` retains three fresh matched
pairs, alternating control/candidate issuance order. The installed wheel build is
`d4c5bc12436c5354c1974b2731c5ccd45ef6e68f35c22023c5dd057e78b0a04e`;
native CKKS implementation identity is
`2abc45fabca231c4a298801842b99887ac742316a9bd211a7cdacb1f5f017134`.
All seven frozen probe/helper/imported source identities matched before and after
workers. No worker failed and no unsuccessful cohort evidence was generated.

Each worker evaluates eight width-eight vectors with sixteen logical head values.
Across six workers this is **96 per-worker logical outputs**, not 96 different
fixtures: all workers replay the same public fixture. Every decoded repetition
and zero-padded coordinate is checked: **55,296 output slots plus 165,888 padding
slots = 221,184 slots**. Maximum absolute output error is
`4.494776259938993e-7`; maximum padding error is `5.643216884297742e-8`, both below
`0.0002`. Every worker reaches exactly zero remaining rescale depth. Integer
exactness remains zero; numeric checking is observed CKKS error, not a universal
error proof. All frame counts exactly match the executable public resource gate.
Watchdog observed **zero swap growth**.

Per-worker byte counts are identical across three repeats:

| Accounted delivery | Serial control | Eight-lane candidate | Control/candidate |
| --- | ---: | ---: | ---: |
| Setup frames | 23,925,248 | 23,925,248 | 1.0 |
| Online frames | 14,746,896 | 1,843,362 | 8.0 |
| Setup plus online | 38,672,144 | 25,768,610 | 1.500746217976057 |

Measured seconds below are **medians per mode**, not median paired ratios:

| Cost for all eight inputs | Serial control | Eight-lane candidate |
| --- | ---: | ---: |
| Setup wall | 0.522372 | 0.479058 |
| Setup process CPU, both roles | 0.520920 | 0.478383 |
| Online wall, excluding oracle validation | 2.803101 | 0.367078 |
| Online process CPU, both roles | 2.726722 | 0.359110 |
| Client online CPU | 0.261350 | 0.026949 |
| Provider online CPU | 2.468367 | 0.332887 |
| Setup plus online process CPU | 3.247642 | 0.833604 |
| Oracle validation CPU, separately charged | 0.018440 | 0.001712 |
| Whole-process peak RSS, bytes | 234,176,512 | 218,152,960 |

Median **paired** ratios are 7.495243 for online wall, 7.566805 for online process
CPU, 9.697948 for client online CPU and 7.392210 for provider online CPU.
Setup-inclusive process-CPU paired ratio is **3.725818**, not the online factor.
Setup includes key/context generation, key delivery and loading in both roles;
its client/provider CPU split is not measured, so setup-inclusive *client* CPU
gain is unknown. Client-online paired ratios range **5.874798–11.785090** and
individual wall timings vary; this three-pair cohort cannot establish a stable
9.7x law or isolate scheduling/frequency/I/O causes of variation. The actual
arithmetic operation reduction is exactly eight: serial/packed counts are
192/24 plaintext products, 208/26 rescales, 168/21 rotations, 184/23 ciphertext
additions, 16/2 ciphertext products/relinearizations and 16/2 modulus switches.

Maximum observed process peaks are 235,520,000 bytes serial and 225,067,008 bytes
packed. Both exceed the 189,255,680-byte native allocation estimate because that
estimate is not whole-process RSS. Admission was 512 MiB; do not relabel the
estimate as a measured peak or derive per-client RSS from the paired-owner process.
Unpadded serialization payload writes and reads are each 61,375,492–61,376,098
bytes serial and 40,928,022–40,934,314 bytes packed; compressed randomized
serializations vary while publicly padded peer frame counts stay fixed.
Physical disk traffic and physical network traffic remain unknown.

### Decision and promotion blockers

This is a positive incremental **same-client finite-state SIMD throughput**
boundary: state survives two encrypted transitions without decryption and eight
independent inputs share arithmetic and fixed ciphertext egress. No ciphertext
size reduction, per-stream autoregressive latency gain or cross-client multi-key
batching follows. Batch-fill delay is excluded. Substantial setup cost narrows
the bandwidth result to 1.50x under the measured eight-input lifetime.

SDK/decoder promotion remains blocked by existing-checkpoint fidelity, real
attention/normalization/SiLU/range contracts, encrypted KV ownership, private token
selection and feedback, deeper finite-depth exhaustion without SEAL bootstrap,
authenticated role transport, circuit privacy/independent cryptographic review
and seven complete whole-response serving metrics. No shared registry, Experiment,
model configuration or SDK availability changed.
