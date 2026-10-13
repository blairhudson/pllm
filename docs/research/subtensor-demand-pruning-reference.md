# Sub-tensor semantic demand pruning: bounded rejection screen

## Decision / payload bound first

Do not promote hypothesis #7 into production. Against existing
`MaskedLinear(prefill_pruning="terminal")`, the only additional remote slice
found in these dense decoder graphs is final-layer Q on earlier prefill rows.
K/V still requires every active row and the entire quantized input width.
One prepared response at 39 input / 8 output tokens uses one Client->Inference
masked input, one Inference->Client reply and one Preparation->Inference
correction per stage lane, four bytes per u32. No cold/repeat workload doubling:

| Representative header | Removed Q coordinates | Reply ceiling | Correction ceiling | Incremental ceiling | Included logical-body fraction | Remote integer MAC retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2, 896 hidden / 24 layers | 34,048 | 136,192 B | 136,192 B | 272,384 B | 0.1935% | 99.8085% |
| Qwen3, 1024 hidden / 28 layers, Q width 2048 | 77,824 | 311,296 B | 311,296 B | 622,592 B | 0.3888% | 99.5979% |
| Llama, 2048 hidden / 16 layers | 77,824 | 311,296 B | 311,296 B | 622,592 B | 0.3603% | 99.6266% |

These are **logical arithmetic ceilings**, not measured response traffic.
Input saving is exactly zero for the split QKV stage. Existing row-residue
encodings can make replies/corrections smaller than u32; no tensor data was
loaded to establish their actual widths. The denominator includes body linear
stages in prefill and seven decode rows, but excludes lookup/head, transport,
control, checkpoint delivery, inventory and other cold artifacts. Its fractions
are illustrative u32-body fractions, not bounds on measured wire percentages.
Existing terminal O/MLP pruning is the comparator, never counted as new savings.
Included single-response comparator bodies are 140,754,944 B / 160,120,832 B /
172,785,664 B respectively. Fractions and integer MAC accounting are unchanged
by correcting directed-link multiplicity; both byte numerator and denominator
were previously doubled, without any corresponding execution/MAC doubling.
Integer work saved per worker: 30,507,008 / 79,691,776 / 159,383,552 MACs,
respectively. Remaining arithmetic stays remote; this does not establish a
whole-decoder FLOP ownership percentage or independent operators.

New client body weights: zero. Full active KV remains required. **<=256 MiB
client gate unestablished**: no RSS, memory allocation or timing measurements.
Small prefill-only increment does not justify new material/frame complexity;
decode savings remain zero. No new admission or cryptographic protocol exists.

The bounded 512-input / 8-output header screen gives absolute reply+correction
ceilings of **3,662,848 B (Qwen2)** and **8,372,224 B (Qwen3/Llama)**.
Input saving remains zero. This larger-context run still loads no checkpoint
tensors and establishes no traffic, performance or RSS result.

## Semantic transformation and blockers

The probe lowers public configurations with installed native semantic compiler,
then inspects ordered grouped stage offsets and widths. It never parses model
node names or weight paths. Backward demand roots include graph output and
every persistent state output. A separate conservative walk reproduces existing
compiler terminal flags. The proposed walk additionally propagates query-row
demand through certified head reshapes, RoPE, attention scores, per-query
scaling/mask, last-axis softmax, attention-values and inverse head reshape.
Key/value operands always widen to all rows. Unknown/layout/state consumers
also widen; query-axis softmax is forbidden. Dense batch-one, sequential absolute
positions and last-valid selection are the bounded header contract. Padding,
arbitrary row selection and custom kernels need separate certification.

Candidate execution: one full-width input opening shared by K/V and terminal Q;
all-row K/V output slices plus terminal-row Q output slice, preserving original
ordered output coordinates. Local Q/attention earlier-row placeholders can be
zero only if no consumer reaches logits, persistent state or exports. Source
compiler currently stops demand at attention/layout operations and admits only
whole groups. Runtime currently binds every grouped operation to one stage,
expects a full-width result, and expands whole terminal groups back to original
shape. Merely setting `terminal_row_only` on QKV is wrong: it destroys KV.

Forbidden consumers: retained/exported Q, all-row attention or hidden outputs,
all-row logits, teacher-forced loss, unknown verification consumers, query-axis
reductions and state/reuse contracts carrying intermediate hidden/query rows.
Tests add persistent Q and opaque consumers as roots and require all-row Q.
All-row hidden export invalidates even existing terminal comparator; probe
rejects such cost comparisons rather than charging unsafe savings.

* **Fresh prefill:** terminal Q eligible only under exact last-valid/output roots.
* **Decode:** one new row, no earlier-row slice to remove.
* **Continuation:** existing compiler clears terminal flags; runtime disables
  terminal elision and enforces complete result shapes. Probe admits no saving.
  Reusing full KV with terminal-only suffix output might permit a separately
  certified slice contract, but continuation APIs may return every suffix logit.
  Do not inherit fresh-prefill eligibility into reuse or restoration.

## Numeric / state gates

Output-coordinate partitioning must retain full input-feature reduction domain,
identical per-input-row quantized values/scales, per-output weight scales and
biases. Cast integer result to float32, multiply activation scale, multiply weight
scale, then add bias in unchanged order. Do not recalibrate inputs by output
slice, partition input features, change RMS/softmax reduction axes, split integer
accumulators into rounded float partials, or change float32 attention reduction
order. Tests include input-domain recalibration and float32 reassociation
counterexamples, and nonzero-bias output slicing. Qwen3 head RMS norms reduce
features independently, never query rows.

Independent generated array interpreter traverses lowered Qwen2/Qwen3/Llama
small two-layer graphs (16 hidden, 4 query heads, 2 KV heads, head dimension 4,
7 rows). Nine deterministic seed/family cases compare every head logit bit
and every active KV coordinate: full execution vs existing terminal comparator
vs proposed output slicing. It uses independent integer linear arithmetic and
synthetic NumPy float32 attention/RoPE; **not production numeric equivalence**.
Production RoPE profile, native quantizer ties, provider accumulators, masks,
continuation execution and checkpoint execution remain untested. Graph roots
and mutation tests are independent from compiler terminal flag calculation.

## Production costs deliberately not built

Would require phase-qualified authenticated output-slice namespace, original
stage/offset/width/row-range bindings, public demand/root/numeric-contract digest,
client/provider agreement, ordered response reconstruction and strict shape
validation. Material issuance needs per-slice one-use corrections, admission,
cancellation and reserve/claim/burn ledger conservation. Splitting independent
stages may duplicate full input openings and masks; sharing an opening needs
explicit protocol authorization. No input-byte saving is assumed. Keeping full
existing reservations only burns Q material and saves online reply lanes;
offline correction ceiling requires demand-aware issuance, not current pruning's
full-row reservation behavior. Extra frames, metadata and admissions are unknown
positive costs, omitted from optimistic ceiling. No low-RSS claim follows from
logical missing rows or placeholder tensors.

## Replay / next command

```bash
PYTHONPATH=. target/frontier-sdk-env/bin/python scripts/probe_subtensor_demand_pruning.py \
  --rows 39 --outputs 8 -o docs/evidence/research-subtensor-demand-pruning.json
PYTHONPATH=. target/frontier-sdk-env/bin/python -m pytest -q \
  tests/test_subtensor_demand_pruning_probe.py
target/frontier-sdk-env/bin/ruff check scripts/probe_subtensor_demand_pruning.py \
  tests/test_subtensor_demand_pruning_probe.py
```

Evidence records configuration/graph/schedule digests, source and owned-file
SHA256 hashes, deterministic array digests, numeric limits and explicit unmeasured
gates. No checkpoint tensor loading, model providers, torch models, build/install,
agents or process timing. Full compiled execution remains **unperformed**.
No `fresh_workers` mode is implemented: reject small ceiling before adding a
generated-checkpoint/provider execution harness. Audit also ran the bounded
`--rows 512 --outputs 8` screen into a fresh ignored `.artifacts/` JSON path; serial
fresh-worker execution would require a new opt-in harness and numeric/state
certificate, not an existing command claimed to support this transformation.
