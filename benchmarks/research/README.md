# Research scorecard configurations

These Python factories return ordinary PLLM `Experiment` objects. Run compared
factories together so the benchmark assigns a shared private cohort salt.

The exact commands, metrics and cost boundaries are documented in
[`docs/content/docs/research/scorecard.mdx`](../../docs/content/docs/research/scorecard.mdx).

| File | Execution scope |
| --- | --- |
| `baseline.py` | Pinned Qwen2.5-0.5B prepared W8A8 control |
| `sota.py` | Exact prepared stack; no assumed public or reusable private prefix |
| `public_prefix.py` | Same stack with a trusted 96-token public-prefix capsule |
| `slalom_baseline.py` | 32+1-token control for verified-adaptation admission |
| `slalom.py` | Slalom-derived trusted-client Freivalds adaptation; current attempt memory-blocked |
| `curl_reference.py` | Bounded native component probe; not a decoder Experiment |
| `combinations.py` | Declared CPU/Metal, placement, protocol, delivery and preparation search axes |
| `search.py` | SDK `BeamSearch` + `benchmark_search`; retains evidence, rejections and exact Python factories |
| `confirm.py` | SDK-selected throughput/byte finalists, saved factories and one fresh matched baseline cohort |

`PLLM_RESEARCH_ARTIFACTS` points to the explicit output directory of
`scripts/build_incremental_sdk_experiments.py`. Public artifact publication and
distribution have their own cost boundary. Generated artifacts belong outside
version control; report JSON and these configurations are retained.

SOTA is selected through matched measurements of compatible SDK combinations.
The scorecard chooses leaders separately per metric and matched cohort. The
`sota.py` filename retains the earlier exact-stack recipe; the measured search
can select different configurations. Search winners are exploratory until a
fresh matched baseline/finalist confirmation. Original-paper performance is
never substituted for a PLLM measurement.
