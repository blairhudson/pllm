# First source-present GKM Qwen response

The explicit Decoder-capacity composition completes one ordinary SDK response
with the pinned Qwen2.5-0.5B-Instruct checkpoint and two separate local worker
processes. The public input `Hello.` renders to four input tokens; fixed-count
greedy execution accepts two output tokens after final native verification.
The API marks the response `incomplete` because it reaches the output cap. The
verified request and benchmark both complete successfully.

This closes the first full-checkpoint functionality gate for this workload.
The source i16/Q11, checked ring64, original float32 lookup, client-local float32,
full-head and full-KV contracts execute without a range rejection. No independent
full-logit/KV or upstream quality comparison runs in this sample. Source-model
quality, broader activation coverage and publication-comparator qualification
remain open. The ten-method comparison still has zero qualified systems.

## Admission and execution identity

- Checkpoint revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Wheel SHA-256: `135fef0c7502e21f82669e1faacdc7a3c1174848f7111d06da4928c0200c7989`.
- The wheel is the cached local build already checked by the
  [Decoder-capacity conformance archive](decoder-capacity.md): 640 installed
  Python tests and 134 native checks plus Clippy.
- All 362 installed package files match before and after the response. All seven
  checkpoint files match their original source lock on post-run hashing.
- The complete compiled bound remains 64 input rows and two output selections,
  with 97 matrices and 1,537,852,928 imported source bytes.
- Frozen preflight requires 11,631,703,100 bytes (10.83 GiB) against 14.87 GiB of
  safe host headroom. Execution rechecks admission against 13.94 GiB of safe
  headroom. Both retain the ordinary 4 GiB host reserve and allocation margins.

The original 8.11-GiB-headroom rejection remains in `decoder-capacity-v1/`.
Later insufficient-headroom retries produced no response measurements. This
successful sample has its own fresh process, cohort and host-admission receipt.
The ordinary response target redirects the role root to its private directory;
configuration-bound runtime digests change, while source, decoder, composition
and complete allocation estimates stay equal.

## One-sample measurements

| Metric | Recorded value | Window or scope |
| --- | ---: | --- |
| Complete request | 143.873 s | Request setup through accepted completion |
| First accepted text | 143.052 s | Online start through final verification |
| Target execution | 166.163 s | Ordinary target, including role startup and cleanup |
| Client peak RSS | 1,916,469,248 bytes (1.78 GiB) | Fresh client process lifetime, including dashboard/supervisor |
| Worker A peak RSS | 1,808,220,160 bytes (1.68 GiB) | Worker process lifetime |
| Worker B peak RSS | 1,804,288,000 bytes (1.68 GiB) | Worker process lifetime |
| Sampled simultaneous client/worker RSS | 4,071,358,464 bytes (3.79 GiB) | 3,036 same-window samples at nominal 50 ms intervals |
| Cold-first CPU, client / A / B | 100.424 / 32.510 / 32.378 s | Client benchmark start and worker births through first response |
| Cold-first aggregate CPU | 165.312 s | Sum of those three measured role windows |
| Covered all-link application bodies | 87,227,888 bytes | Setup, online and teardown |
| Online application bodies | 44,523,072 bytes | Prefill 34,150,560 + decode 10,372,512 |
| Observed host swap growth | 0 bytes | Ordinary 250 ms host-pressure samples |

The simultaneous RSS is a sampled sum, not a sum of independent lifetime peaks.
Samples can miss short peaks or count shared pages twice. OS file cache remains
outside process RSS. The one observed client peak fits the 2 GiB Medium ceiling,
but the conservative client admission is 3,962,826,300 bytes (3.69 GiB). This does
not qualify an SDK Medium memory cap, a replicated tier result or a memory gain.

Request-end verification buffers both outputs. Decode throughput is therefore
unavailable; text-buffer release is not model decode speed. The native runtime
checks pass, with 974 worker operation calls and zero recorded plaintext prompt
or token-ID bytes. Those counters do not establish cryptographic security or
independent operators. Both workers share one host/operator.

There is no WAN pacing in this sample. Covered application bodies exclude HTTP
headers/framing, monitoring, checkpoint distribution and physical wire. The
report's 100/40 Mbps values are analytical floors only. No matched control or
independent confirmation runs here, so these costs establish no comparative win.

## Retention and read-only replay

`qwen-response-v1/` retains 14 hash-bound files: the original report, runner,
Experiment, source/runtime/native contracts, preflight, input and installed-file
receipts, execution log, post-run checks and capture program. The capture records
zero visible workers using this interpreter/module after completion; 221 unrelated
processes deny inspection. This is a process-table snapshot, not historical child
PID attestation. Original absolute paths remain as provenance.

From the repository root:

```sh
python3 -I -B paper/artifacts/gkm-sdk/verify_qwen_response.py
```

Replay validates the archived conformance link, original bytes, installed build,
source and geometry, fresh admission, runtime acceptance, phase/role body
conservation, resource windows and unavailable decode throughput. Negative
controls in `tests/test_gkm_qwen_response_archive.py` reject changed sources or
builds, rejected admission, omitted workers, swap growth, altered metrics and
promotion to independent quality or operator evidence.

For a new execution, use the retained `runner.py` with the same installed wheel
and dependencies. Its `prepare` mode accepts a fresh output directory and checks
ordinary admission without reading tensor values. Only an admitted preparation
may run `execute`, which rechecks live admission. Relocate source/interpreter
paths as needed and retain the new cohort separately. This read-only verifier
never runs a model or treats a saved preflight as a live execution permit.
