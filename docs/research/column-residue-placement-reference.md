# Input-column residue threshold placement: optimistic certificate

Default probe replays source-locked public histograms for 96 original W8A8
Qwen2.5-0.5B body stages. No checkpoint import, downloads, providers, protected
material, CPU/RSS timing or private traces. This is not Low qualification.

## Exact reconstruction

Keep original activation quantizer/domain `[-127,127]`. For selected public input
columns L, compute local integer `l_j=sum_L W_ji*x_i`. Remaining public sum
`S_j=sum_remote abs(W_ji)` gives original compiler rule
`h_j=max(1,bit_length(127*S_j)+1)` and centered range
`[-2^(h_j-1),2^(h_j-1)-1]`. Fresh uniform remote input masks r and independent
output masks s give correction `W_remote*r-s`. Upload remote coordinates
`(x-r) mod 2^max(h)`, receive `W_remote*(x-r)+correction mod 2^h_j`, add s and
center to recover exact remote integer. Add local integer **before** original
dequantization, original activation/weight scales and original bias, applied once
in original operator order. Adding independently rounded float partials is invalid.
Zero remote rows retain one-bit compiler representation; fully local groups omit
all three remote frames. Input -128 is rejected, not silently admitted by 127S.
Weights include -128; hence removed mass bounded by 128k, not 127k.

Original compiler/installed protocol need a new source-bound column contract,
correction schedule and protected lifecycle. Toy oracle proves arithmetic only;
no new distributed security or full-logit/KV execution claim.

## Certificate and identical-budget control

Histogram width h implies `S >= ceil(2^(h-2)/127)` (h=1 means S=0). Any k selected
columns remove at most 128k in each row. Substitute this smallest possible S,
subtract 128k and rederive widths. This simultaneously optimistic relaxation may
cross incompatible rows and is **not** an achievable column recipe. Remaining
request uses maximum remaining lower width, corrections and replies use packed
sum of per-row lower widths, independently rounded up to bytes.

Specified artifact layout charges n*k raw i8 bytes, four bytes per selected column,
64-byte artifact header; original scales/bias remain common baseline delivery.
Per-executed-row framing model: three 64-byte frames; partial columns add 16-byte
map-binding overhead to each. These are explicit modeled costs, not measured
HTTP framing. File ownership and native i8 weight bytes recorded separately;
native scratch, page cache, whole-client peak and CPU remain null. No compressed
weight-delivery lower bound is claimed. Different codecs require repricing.

Global budget floor(total body MACs/10) independently limits prefill/decode.
Whole-group 0/1 knapsack enumerates all layer/semantic groups exactly for same
raw artifact/cache/frame model. Its weight ownership stays below 256 MiB; this
does **not** account for existing whole-client allocation. Column upper bound
is min(sum independent stage maxima, budget*maximum net-saving/MAC ratio),
rounded down. Whole-group comparisons therefore bound possible additional gain,
not measure it. Native/file copies and existing bundle consume headroom; unknown
full-client costs prevent promotion even if optimistic upper bound survives.

Report shows 150 prefill and seven executed decode rows separately, plus combined
157 rows; no terminal pruning, prefix cache or measured historical all-link ledger
is borrowed. Horizons 1/4/16/64 use distinct executions sharing selected public
weights. Empty cache pays artifacts once; preowned cache excludes prior sunk
delivery identically for both strategies. Other common body/artifact/control
charges cancel in savings, not claimed absent. Separate phase optima are upper
envelopes and cannot be combined into one recipe.

Ideal bit-packed ingress-only raw repayment is `ceil(8*n/request_bits)` executed
rows per column, before maps/headers and per-row byte rounding. Exact modeled break-even for any actual selection is
`ceil(delivery/saving_per_row)` when saving positive; otherwise no repayment.
Histograms alone cannot select public columns or prove held-out real-stage gains.

At 157 cold executions, ingress-only removal cannot repay any nonempty selection
in specified raw layout while request width and output costs remain unchanged.
Exact per-stage search enumerates every admitted k, including byte rounding;
best nonempty net is recorded, and no placement wins this restricted screen.
Residue/request threshold crossings are therefore essential, not established by
histograms. Their relaxation is very loose: it can attribute row bounds near
every threshold simultaneously, impossible to validate without correlations.

Retained result: best nonempty cold ingress-only net **-493 bytes** (one down
column: 964 delivery bytes, three saved ingress bytes per row). Its exact modeled
repayment is **322 rows**, not 157. Ideal raw thresholds range 276–3,243 rows.
At cold horizon one, whole-group optimum saves 15,260,684 prefill bytes, zero
decode bytes, or 17,642,826 combined bytes; combined recipe owns 35,785,216
artifact bytes and exactly 10% of body MACs. Column saving upper ceilings are
265,729,704 / 349,416 / 280,949,398 bytes respectively. Their enormous gap from
feasible whole-group control demonstrates histogram insufficiency, **not** a
predicted win. All 1/4/16/64-horizon comparisons remain unresolved.

## Validation and parent-only next gate

Bounded tests include 3,000 independent integer outputs, every subset, activation
endpoints, -128 weights, zero rows, threshold boundaries, exhaustive subset bound
checks and independently enumerated whole-group knapsack controls. Source and
own implementation files hashed. Default evidence is deterministic; hashes do
not include evidence itself (avoids self-reference).

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. tests/test_column_residue_placement_probe.py
target/frontier-sdk-env/bin/python scripts/probe_column_residue_placement.py --output target/column-residue-replay.json
```

Only parent, after concurrent workers finish, may run guarded offline scan:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 target/frontier-sdk-env/bin/python scripts/probe_column_residue_placement.py --scan-source --output target/column-residue-source-fresh.json
```

Opt-in uses existing fresh_workers lock, memory/swap guard, 6 GiB admission and
600-second timeout; checkpoint fixture can own full body, so never launch during
concurrent workers. Fixed public descending column-L1 rule (index tie-break) is
unchanged on held-out odd layers; even layers tagged calibration. Scan checks
original weight hashes, compiler histograms and certificate inequality at fixed
1-column/1%/5%/10% prefixes. It only refines geometry, not global optimum, serving
or Low admission. Next gate: actual compressed selected artifacts and global
column/group allocation under measured whole-client headroom; then independent
exact logits/KV and matched prefill/decode remote fractions, cold/repeat traffic,
CPU/native ownership and fresh-process <=256 MiB peaks. No private optimization.

**Verdict:** histogram certificate is bounded screen, not evidence of H2 win.
Real column correlations and whole-client costs remain parent-owned gates.
