# Publication lifecycle observations

Status: **bounded local cold measurement capability verified**. This observer
supports the framework study's phase and role accounting. It does not supply
ten complete method implementations or a serving ranking.

## Measurement contract

`scripts/benchmark_publication_lifecycle.py` wraps the ordinary installed
`BenchmarkTarget.response` path. It selects maintained prepared execution,
MLP tiling, and parallel/residue coupled boundary execution. The first two use
two provider processes; the third uses four. The client owns the SDK,
dashboard, supervisor, relay, and state audit.

Each party has shared native 100 Mbps download and 40 Mbps upload limits.
The local emulator adds no RTT or loss. Provider CPU and RSS quotas are unset;
the extra boundary parties have extra aggregate access capacity. The profile is
explicitly `resource_unnormalized_per_party_100_down_40_up`.

The observer records these phases on one monotonic client clock:

| Phase | Boundary |
| --- | --- |
| Public setup | Observer start through request admission, excluding explicit inventory issuance. |
| Request setup | Admitted request work outside explicit issuance and semantic execution. |
| Private preparation | Inventory issuance calls. |
| Prefill | Initial semantic forward through its audited first selection. |
| Decode | Later semantic forwards through their audited selections, including an unemitted EOS. |
| Unclassified | Emission and inter-step gaps outside the explicit semantic phases. |
| Cleanup | Terminal EOS selection or cap emission through the post-serialization exit observation. |

Ordinary first/last text events separately determine first-text delay and
`N−1` decode throughput. Semantic decode time is not emission-interval time.
The frozen public prompt retains its original bytes. Token counting applies
the same trimming and context validation as the dashboard request.

### Memory and owners

Each sample records process identity, role, host, RSS, and its read interval.
PID plus process creation time distinguishes process incarnations. Independent
provider startup receipts bind the imported package and native extension.
Observed helpers have explicit joint ownership where applicable.

For each phase, the report gives per-instance peaks, instance counts, and
simultaneous sums by role, host, all non-client roles, and complete deployment.
It takes each aggregate maximum from a single sample vector. It never adds
independent lifetime peaks. Read skew and sample intervals remain visible.

The sampler uses a timeout-derived interval of at least 50 ms, with additional
boundary observations. Trace and provenance admission add 64 MiB and 16 MiB,
respectively. Rows stream to private disk; the controller replays them after
worker exit. The 1,200-second trial deadline retains capacity for shutdown and
boundary samples without exceeding 32,768 rows of at most 8,192 bytes each.

Three memory windows remain distinct:

1. Sampled phase/role RSS covers the observer window.
2. Ordinary `process.peak_rss` keeps its process-birth-through-target meaning
   and its existing observed-tier rule.
3. A separate `RUSAGE_SELF` receipt extends through the post-serialization exit
   observation. Later receipt serialization and interpreter teardown are outside it.

Sampled sums can miss short peaks and count shared pages more than once.
Provider lifetime high-water, device memory, filesystem cache, complete process
lifetime, and remote-host measurements remain unknown.

### Traffic and retirement

Directed TCP-payload counters conserve bytes against per-party and phase totals.
Ordinary benchmark snapshots reconcile within their measured read intervals.
HTTP/WebSocket framing belongs to this stream scope; physical headers, ACKs,
retransmissions, checkpoint distribution, and Internet latency do not.
Application-body attribution remains partial.

Successful retirement requires joined dashboard, client, sampler, observer, and
relay owners, closed role groups/listeners, an empty relay, and no live children.
The actual execution thread remains owned through tokenizer teardown. Escaping
errors and relay failures invalidate completion.

## Durable evidence

Each trial gets an exclusive `.traces/<nonce>/` directory before launch. Raw
samples, final state, ordinary payload, parent/provider receipts, and a complete
hash/size manifest remain after temporary-runner cleanup or result construction
failure. Ordinary result details retain a bounded reference and summary rather
than the full trace. Existing outputs, companions, and symlink substitutions reject.

Parent acceptance and `load_trace(reference)` use the same qualification path.
It binds the nonce, configuration, manifest, receipts, sample hash/size/count,
timing, counters, source identities, and ordinary result. Bounded nonblocking
reads reject nonregular files and symlinks before parsing. The same file descriptor
supplies bytes and hashes, with identity and mutation checks.
Explicit diagnostic recovery returns an unqualified, incomplete trace on failure.
These checks establish local evidence consistency, not remote attestation.

## Verified tiny cohort

The third attempt ran three fresh clients on a generated checkpoint with
103 input tokens and two emitted outputs. All state bindings agree. Its
289 samples contain 11 process incarnations across the three-/five-role
configurations, with zero observed swap growth. Independent replay recomputes
every phase/role/host simultaneous peak and conserves the recorded stream bytes.

The first attempt failed final receipt generation. The second failed prompt-size
reconciliation. Both remain rejected and archived. Earlier inline-v1 results
retain their historical semantics and gain no durable-archive guarantee.

The regression suite passed **175 tests**. It covers capacity, large traces,
teardown/errors, manifest and receipt substitutions, counter chronology, unsafe
files, read races, token counts, and actual exit behavior. The frozen archive
contains 144 original/source files totaling 4,401,845 bytes; its read-only replay
checks all three trials. These are measurement controls, not model-quality or
performance results.

Archive: [`paper/artifacts/publication-lifecycle/2026-10-17-tiny-v3/`](../../paper/artifacts/publication-lifecycle/2026-10-17-tiny-v3/README.md).

Verified commands, from the repository root:

```sh
paper/artifacts/clean-build/2026-10-16-local-v5/work/venv/bin/python -B scripts/audit_publication_lifecycle.py
paper/artifacts/clean-build/2026-10-16-local-v5/work/venv/bin/python -B -m pytest -q -o pythonpath= tests/test_publication_lifecycle.py
```

The completed smoke used:

```sh
paper/artifacts/clean-build/2026-10-16-local-v5/work/venv/bin/python -B scripts/benchmark_publication_lifecycle.py --tiny --replicates 1 --output target/research/publication-lifecycle-archive-2026-10-17-v3/cohort.json
```

That output is write-once. A new execution needs a new output path and fresh
resource admission. Real-checkpoint cohorts, matched provider budgets, warm and
concurrent requests, and complete ten-method measurements remain open.
