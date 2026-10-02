# Selected-network gateway and ordinary benchmark evidence

Evidence label: 2026-10-01. Captured locally on 2026-10-02.

Slice C connects selected deployments to the existing `gateway` and `benchmark
run` command families. Results are in
[`network-planner-benchmark-2026-10-01.json`](network-planner-benchmark-2026-10-01.json).
This is a controlled loopback diagnostic, not independent-operator or WAN evidence.

## Executed fixture

From the repository root, the following command was executed successfully:

```sh
.venv/bin/python examples/benchmarks/network_scenarios.py \
  --output docs/evidence/network-planner-benchmark-2026-10-01.json
```

The script generates a checkpoint, source lock, strict network/request/snapshot/plan
JSON files and a private prompt file in a temporary directory. It starts two real
HTTP provider applications, each hosting two installed role instances, and four
HTTP shaping proxies. All gateway subprocesses, HTTP listeners and temporary
files are cleaned up on exit. Credentials are generated in memory and passed by
configured environment-reference names; no credential values enter JSON.

Both scenarios use the same generated checkpoint contents, W8A8 body, one layer,
hidden size 32, intermediate size 64, four attention heads, two KV heads and head
dimension 8. Execution bounds are one response attempt, 32 input tokens and two
new tokens. Actual authoritative usage is 22 input tokens and two output tokens.
Sampling is explicitly greedy (`--temperature 0`). Offers and snapshot/link
fixtures have finite 60-second validity. Controller role capacity is two sessions.

Four installed instances advertise actual role allowlists: `a-east`/`a-west`
host `worker_a`; `b-east`/`b-west` host `worker_b`. The trusted client offer remains
local. Discovery replaces provider placeholders with authenticated live offers.
All four ordered feasible host pairs execute the same pipeline and source lock.

Directed client-to-provider and provider-to-client rates are 20,000,000 B/s for
the preferred region and 200,000 B/s for the other region. Proxies delay actual
request/response bodies by their lengths divided by the configured directed
rates. They record both requested sleep durations and observed monotonic sleep
intervals, plus body lengths and request/error counts. They never retain body
contents, activation frames, capacity tokens or authentication headers.

The public network schema admits these bandwidth settings as **estimates**.
The separate proxy ledger records controlled observations; it does not relabel
estimates as measured native planning evidence.

## CLI commands exercised with generated files

`FIXTURE` below denotes the script's live temporary directory for one scenario.
These argument forms were executed against the generated files and running
services, rather than printed as suggested commands:

```sh
.venv/bin/python -m pllm --format json --dry-run gateway \
  --plan "$FIXTURE/plan.json" --network "$FIXTURE/network.json"
.venv/bin/python -m pllm --format json --dry-run gateway \
  --request "$FIXTURE/request.json" --network "$FIXTURE/network.json"
.venv/bin/python -m pllm --format json --dry-run benchmark run \
  --plan "$FIXTURE/plan.json" --network "$FIXTURE/network.json" \
  --prompt-file "$FIXTURE/private-prompt.txt" --max-output-tokens 2 --temperature 0
.venv/bin/python -m pllm --format json benchmark run \
  --plan "$FIXTURE/plan.json" --network "$FIXTURE/network.json" \
  --prompt-file "$FIXTURE/private-prompt.txt" --max-output-tokens 2 --temperature 0
.venv/bin/python -m pllm --format json benchmark run \
  --request "$FIXTURE/request.json" --network "$FIXTURE/network.json" \
  --compare-feasible --prompt-file "$FIXTURE/private-prompt.txt" \
  --max-output-tokens 2 --temperature 0 --warmups 1 --repetitions 3 \
  --capture-output-digest
```

The fixture also launches real `gateway --plan ... --network ... --port PORT` and
`gateway --request ... --network ... --port PORT` subprocesses. Gateway credentials
come from `PLLM_GATEWAY_API_KEY`. Each mode completes Responses and streaming Chat
Completions calls with two authoritative output tokens and releases capacity.
Dry-run tests verify zero reservations and zero consumed controller attempts.

Selection modes mutually exclude `--experiment`, `--plan` and `--request`.
Network selection requires `--network`; gateway local/manual-provider and
quantization/runtime overrides are rejected. Benchmark temperature omission
preserves the SDK's effective 0.8 default, and reports archive effective sampling.
Output fingerprints require explicit public-task opt-in.

## Matched results

Each scenario measures four feasible assignments in deterministic shuffled order,
with one warmup and three repetitions per assignment. The ordering seed controls
only public benchmark order; it is never a protocol mask or sampling seed and is
not included in the JSON artifact.

| Scenario | Selected worker hosts | Selected median admission-through-release | Slow/slow median | Measured selected-plan latency regret |
| --- | --- | ---: | ---: | ---: |
| east-fast | a-east, b-east | 0.3323 s | 0.9007 s | 0.0000 s |
| west-fast | a-west, b-west | 0.3251 s | 0.9105 s | 0.0000 s |

Both comparisons pass existing workload-cohort rules: actual body fingerprint,
authoritative input/output counts, output cap, observed cache state, private salted
prompt identity and effective sampling match. Public output-text digests match
across all measured controls. Output text and generated token IDs are not exported.
The selected assignment changes when directed bandwidth settings change.

Warmups do not imply a warm disposable client bundle. Existing bundle cache scope
includes rotating lease credentials; every archived run observes a cache miss and
is correctly marked cold. Provider model artifacts remain loaded throughout the
fixture. Each response still opens fresh protected sessions under a fresh lease.
No warm-client-cache speedup is claimed.

Latency regret uses matched median **measured admission-through-release durations**,
excluding planning and provider model startup. It is not derived from the sum of
transfer estimates, and is not a full-cost regret claim.

## Prediction and measurement scopes

Across each comparison's 16 attempts, the bounded arithmetic transfer estimate
sums to 4.4369 seconds. Observed online shaping intervals sum to 3.0501 seconds for
east-fast and 3.0652 seconds for west-fast; observed-minus-predicted differences
are -1.3868 and -1.3717 seconds respectively. Requested online shaping time is
2.7415 seconds in each scenario. Observed intervals include sleep scheduling error.

These differences compare **bounded arithmetic-array estimates** with actual
serialized online bodies and controlled sleep intervals. The request uses fewer
input tokens than the compiled bound; serialized framing adds different terms.
Neither quantity is a critical-path response-latency estimate. RTT, overlapping
work, computation and other response phases are excluded from this transfer-only
comparison. Whole-response latency is measured independently.

The ordinary `BenchmarkRun` record and existing loopback report/cohort helpers are
reused. Additional audit fields record:

- Selected public role-to-host assignments and original typed predictions.
- Covered SDK setup/online/teardown/bundle body counters, counted once per edge.
- Client process CPU deltas, explicitly scoped to the shared process.
- Authenticated existing worker process CPU/RSS samples before and after execution.
  Worker processes may be shared; samples are never summed into a whole-response
  CPU claim.
- Actual client body ownership/native snapshot counts before disposable client
  teardown, using the same helper as the ordinary dashboard.
- Closed lease status and observed public-bundle cache state.

The proxy ledger additionally observes 32 reserve, 32 arm and 32 release calls per
comparison, with their request/response body lengths. Successful fixtures have no
non-200 controller responses. Generic B `ExecutionLease` exposes no admission
counter interface, so ordinary benchmark controller body/count/retry fields remain
`null`; proxy observations stay separately scoped to this controlled fixture.
Full wire bytes, aggregate full-response CPU and full-cost regret remain `null`.
These missing terms disqualify full-cost comparisons.

All provider capacities return to zero. Exported records contain no prompt text,
output text/token IDs, seeds, masks, plaintext activation frames, credentials or
private controller credential hashes.

## Verification

Executed focused suite:

```sh
.venv/bin/pytest -q \
  tests/test_network_benchmark.py tests/test_network_gateway.py \
  tests/test_benchmark_sampling.py tests/test_benchmark_cli.py \
  tests/test_sidecar_gateway.py tests/test_runtime_servers.py tests/test_cli.py \
  tests/test_network_planning.py tests/test_native_network_placement.py \
  tests/test_network_execution.py
```

Result: **316 passed**. This covers the existing native planner/oracle and B
admission tests, all three A local selected graphs, live CLI gateway modes,
official OpenAI SDK Responses/Chat calls, streaming disconnect after partial
tokens, fresh one-use attempts, early token-budget rejection, capacity cleanup,
legacy gateway mocks and existing benchmark behavior. Cancellation tests observe
one admitted attempt per response and no automatic retry.

Focused Ruff checks and `git diff --check` pass. The existing
`.venv/bin/python examples/benchmarks/network_planning.py` SDK/ordinary benchmark
parity demonstration also completed successfully. No staging or commits performed.
