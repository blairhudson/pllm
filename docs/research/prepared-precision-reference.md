# Frozen prepared-precision screen

## Contract

`benchmarks/research/prepared_precision.py` freezes four ordinary Experiment
choices before scoring: W8A8, W4A8, W8A4 and public per-channel equalized W8A8.
The shared compiled local-clear quality driver compares them against a streamed
upstream float32 reference. It scores 12 held-out prefills and three same-token
decode steps per prompt. The first candidate's greedy trajectory fixes decode
inputs for every candidate and the reference.

Equalization uses the existing eight public calibration sequences and fixed
fitting procedure. The held-out file is
`benchmarks/research/data/prepared-precision-quality.json`; none of its 12 prompts
appears in calibration. The profile is frozen before quality execution, with
source-lock, calibration and profile digests retained. No held-out fitting or
private prompt-dependent calibration occurs.

The local-clear diagnostic explicitly selects resident delivery for every scored
Experiment. This avoids identifying a resident import with the prepared factory's
paged-client execution. It does not launch provider roles or measure transport.

## Admission and public compilation

The initial Qwen3-4B profile fit rejects before model-value loading: its allocation
estimate is **9.51 GB** against **8.12 GB** post-reserve headroom. That rejection
is retained in `docs/evidence/research-prepared-precision-qwen3-profile.json`;
no Qwen3 calibrated quality result follows.

The separately pinned Qwen2.5-0.5B source passes the same admission. Public fitting
costs **21.78 process CPU seconds** and **22.69 elapsed seconds**, producing a
**729,527-byte** profile for 96 stages. Diagnostic lifetime peak RSS is 1.47 GB,
including compilation and imports. No swap growth is observed. These are
publisher costs, not response costs. The fit report is
`docs/evidence/research-prepared-precision-qwen25-profile.json`.

## Held-out result

| Frozen choice | Prefill selections | Decode selections | Worst prefill / decode logit error |
| --- | ---: | ---: | ---: |
| W8A8 | 10/12 | 35/36 | 3.626 / 3.756 |
| W4A8 | 1/12 | 21/36 | 14.913 / 14.681 |
| W8A4 | 0/12 | 0/36 | 22.950 / 27.883 |
| Equalized W8A8 | 11/12 | 32/36 | 2.179 / 3.293 |

Equalization improves checked prefill agreement and worst logit error but loses
three decode selections relative to ordinary W8A8. Lower precision is much worse
on this cohort. None earns a default change or a same-quality bytes/compute
ranking. Changed numeric identities must retain separate benchmark contracts.
The earlier SmoothQuant cohort used different held-out prompts and is not pooled
with this result.

The complete report, exact Experiment documents, source and token-cohort
commitments, phase scores and environment are retained in
`docs/evidence/research-prepared-precision-qwen25-quality.json`. The quality
diagnostic peaks at 2.31 GB process RSS and observes zero swap growth. Its
375-second elapsed duration is diagnostic work, not inference latency; other
development checks ran concurrently. This is a narrow numeric screen, not
representative generation quality, protected execution or provider performance.

## Reproduction

The existing prepared-frontier artifacts supply the ordinary base factory.
Public profile publication uses the standard content-addressed equalization
store. Both commands require new report paths; the quality command verifies the
fit report's source and calibration commitments.

```sh
PLLM_RESEARCH_ARTIFACTS=target/prepared-frontier-artifacts \
.venv/bin/python -m benchmarks.research.prepared_precision fit \
  --experiment benchmarks.research.prepared_frontier:serial \
  --output target/evidence/prepared-precision-fit-reproduction.json

PLLM_RESEARCH_ARTIFACTS=target/prepared-frontier-artifacts \
.venv/bin/python -m benchmarks.research.prepared_precision quality \
  --experiment benchmarks.research.prepared_frontier:serial \
  --profile-report target/evidence/prepared-precision-fit-reproduction.json \
  --output target/evidence/prepared-precision-quality-reproduction.json
```

Use the `qwen3_memory_frontier:paged` factory and its separately built public
artifacts to reproduce the larger-model admission check. Available headroom is
host-dependent; a historical rejection is not a universal capacity verdict.
