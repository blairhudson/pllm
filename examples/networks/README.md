# Authenticated two-party offset execution (slice B)

Two separate HTTP party processes reuse PLLM's installed offset role adapters,
compiled decoder checks and SDK. Both processes run on one machine, under one
owner: different operator labels are declarations, **not deployment privacy proof**.
Offers explicitly report `independence_verified=false`. No attestation is enabled.

## Run

```sh
python examples/networks/two_party.py prepare --root /tmp/pllm-network-example
export PLLM_NETWORK_A="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export PLLM_NETWORK_B="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
pllm network inspect /tmp/pllm-network-example/network.json
```

Start each party in a terminal inheriting those environment variables:

```sh
pllm serve party --network /tmp/pllm-network-example/network.json --party /tmp/pllm-network-example/party-a.json --port 8101
pllm serve party --network /tmp/pllm-network-example/network.json --party /tmp/pllm-network-example/party-b.json --port 8102
```

Then:

```sh
pllm network parties /tmp/pllm-network-example/network.json
python examples/networks/two_party.py run --root /tmp/pllm-network-example
pllm network drain /tmp/pllm-network-example/network.json --party a
pllm network leave /tmp/pllm-network-example/network.json --party a
```

Credential values remain local; exported JSON contains environment **names** only.
Offers come from authenticated installed runtimes. HTTP is permitted only on
loopback; other configured origins require HTTPS. A TLS terminator may front the
party ASGI application.

## API and schema contracts

- `NetworkSpec` retains static `pllm.network_spec.v1` / `backend="local"` bytes.
  Live `pllm.network_spec.v2` uses `backend="http"`, explicit `PartyTrust` roots,
  allowed operator IDs, origins and environment credential references.
- `Deployment.network(network_id=..., network_spec_digest=...,
  snapshot_digest=None)` writes explicit `pllm.experiment.v3`. Network intent has
  no `root` field; Python `root` is `None`. Explicit null snapshot means planning
  from fresh offers; a digest locks a particular planning snapshot. Local v2
  documents retain their existing canonical bytes and composition identity.
- `PartySpec` fixes exact installed Pipeline parameters (including CPU threads),
  checkpoint source lock, role instances, workload and aggregate resource ceilings.
  Each restart generates a new epoch. No advertisement installs code.
  Installed offset session limits are at most 64 input tokens, 32 output tokens,
  and 8,192 stage calls; larger configured workloads fail admission.
  Artifact policy requires an already installed checkpoint directory; hosts do
  not implicitly fetch model files during admission.
- `discover(network, credentials=None)` authenticates current offers. Optional
  `credentials` maps configured environment **names** to values; omitted mapping
  reads the environment. No probe, prompt, private state or prepared material is
  sent to discovery.
- `open_execution(result, network=network, credentials=None, ttl_seconds=60)`
  returns an opaque, non-copyable, non-serializable `ExecutionLease`. TTL is 5–300
  seconds and bounded by each party. `OpenAI(execution=lease)` uses the existing
  executor; input/output budgets are checked before provider requests. Initial
  network leases allow exactly one bounded response attempt (`budget.requests=1`).
- `await async_open_execution(...)` plus `async with lease` supports
  `AsyncOpenAI(execution=lease)` with shielded partial-admission and exit cleanup.
  `await async_discover(network, credentials=None)` provides read-only async discovery.
- Capacity tokens differ per attempt and role. Middleware accepts armed, unexpired
  scoped tokens, then privately translates to the existing adapter bearer key.
  Client never receives the static key. Existing model, composition, numerical,
  decoder-plan, tensor and one-use session checks still run.
- Reserve order is deterministic `(party_id, role_id)`. Failed/ambiguous reserve
  or arm releases all acquired roles. Unreachable controllers expire independently.
  Retrying creates a fresh attempt, tokens and sessions; no material transfer.
- `drain` refuses new reservations while admitted work finishes. `leave` returns
  `draining` until leases release/expire, then `left`.

## Optional directory

Set `directory_origin` and `directory_credential_env` in the **network spec before
generating v3 intent**; changing the network changes its digest. Start:

```sh
pllm serve directory --network NETWORK.json --port 8001
```

Parties renew/withdraw membership using their configured credentials. Registration
pulls installed offers from approved party origins instead of accepting capability
JSON. Directory offers expire; it stores no prompts, private KV, roots or tickets.
Live discovery reads authenticated membership and rechecks actual party offers.
Directory membership never promises capacity or physical independence.

## Supported scope and remaining integration

Live network admission supports CPU `TwoOnlineOffsetCpu` compositions admitted by
the native complete decoder scheduler, with whole logical `worker_a` / `worker_b`
instances. Prepared/verified-prepared and client-only graphs remain supported by
slice A's local planner/launcher. They are not selectable from these live offers;
live opening rejects other graphs with `UNSUPPORTED_NETWORK_GRAPH`. Prepared
network admission requires scoped preparation→inference push authorization and
reserved-inventory burn wiring before it can be enabled.

Coordinator C can use the above admission interface for benchmark/gateway selection.
Their new parser flags and orchestration are outside this example. Saved decision
costs remain native/scenario estimates; CPU, full wire, measured peak memory and
separate-host independence evidence remain unknown.

Integration note: slice A's existing search uses coarse capabilities. For hosts
exposing only one worker instance, placement search still needs a live-only
`role_ids` eligibility filter; draining offers also need an `accepting` filter.
Live admission already rejects both cases before reserving/executing. This example
installs both virtual worker instances on each party, so the selected two-party
assignment is executable without changing slice A's selection logic.

Acceptance regressions: `tests/test_network_execution.py` and
`tests/test_experiment_v3.py`; existing A tests remain in
`tests/test_network_planning.py`. They include real two-process HTTP SDK generation
and cancellation, stale/forged commitments, expiry/session burn, capacity races,
partial commit, drain/leave, directory membership, asynchronous cleanup, and dry-run
non-mutation. These are functionality tests, not independently operated deployment
evidence.
