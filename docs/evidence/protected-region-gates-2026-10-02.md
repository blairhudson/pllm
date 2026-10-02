# Fresh-response protected-region reduction gates

## Reproduce

From repository root, using only the pinned config already in the shared Hub cache:

```sh
.venv/bin/python scripts/probe_region_contract_cost.py --max-output-tokens 8 --reduction-gates --summary
.venv/bin/python scripts/probe_region_contract_cost.py --max-output-tokens 32 --reduction-gates --summary
```

No weight import, protected execution or material issuance occurs. The script
checks the semantic plan, schedule and W8A8 composition against the retained
39+8/39+32 prepared controls before pricing any target. Source is pinned
Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775`.

## Decisions

All figures below are decimal MB of **covered bodies**, not full wire. Targets
require both online and all-link costs to fit. No candidate passes numeric,
privacy, compute or execution admission.

| 39+32 candidate | Known online floor | Known all-link floor | 25% / 50% reduction | 10× |
| --- | ---: | ---: | --- | --- |
| Client attention / two-worker MLP, unknown correlations | 21.07 | 21.07 | Inconclusive | Veto |
| Same boundary plus current quadratic-reference keys | 21.07 | 284.28 | Veto | Veto |
| Resident workers, 24-bit sources, unknown correlations | 18.67 | 18.67 | Inconclusive | Veto |
| Same resident layout plus current quadratic-reference keys | 18.67 | 281.88 | Veto | Veto |
| Hypothetical 12-bit resident sources | 9.64 | 9.64 | Inconclusive | Inconclusive |

The 39+8 cohort reaches the same decisions. Current quadratic key bodies alone
are 172.97 MB at 39+8 and 263.21 MB at 39+32; they only implement a bounded
numerator reference. Charging them is a veto of this specific reference-derived
construction, not a lower bound on every future correlation method.

Unknowns are explicitly retained: private per-row scales, max-absolute dynamic
quantization, reciprocal and ties-to-even/range handling, complete SiLU×up,
RMSNorm inverse square root with epsilon, attention/softmax, KV share maintenance,
both parties' extra one-use material, source distribution/storage, framing,
admission/replay/cancellation, client compute, peak memory and latency. Unknowns
never become zero and no generous byte budget can turn this report into a pass.
Token feedback here reconstructs final hidden state at the client, applies its
public head and freshly shares next-token embedding; it is not a private remote
selection/feedback implementation.

## Next direction

Fresh-prompt **25%/50%** improvements remain worth testing only with a concrete
new protected-MLP mechanism and numeric contract. At 39+32 the known two-worker
boundary leaves 113.15/68.41 MB total for every unpriced cost, respectively;
online headroom is 64.08/35.70 MB. Existing quadratic keys do not fit either.

Fresh-prompt **10×** remains parked: the ordinary 24-bit boundaries already fail.
The hypothetical 12-bit layout leaves only 1.72 MB online before norms, attention,
conversion and control, and has no validated widening/rounding contract. This
does not authorize implementing another narrow-ring decoder.

Continue measured exact conversation reuse, demand refill and public artifact
locality while retaining prepared inference as control. A new protected method
must first provide a complete costed operator/material schedule, then held-out
numeric parity and matched aggregate compute; no transport rollout before that.

Raw reports: `protected-region-gates-qwen25-{8,32}-2026-10-02.json`.
