# Run the local development benchmark

Run the real PLLM roles on one machine and understand what the result does and does not measure.

[View canonical HTML](https://pllm.run/learn/start/first-local-benchmark/)

Document ID: `pllm.docs.start.first-local-benchmark`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:45fedca8775a74b50f08153e721f677d3ee5b6138d89cc97b85aff0832e3fa52`

Run the real local client, preparation, and inference roles:

```bash
pllm dev dashboard --model Qwen/Qwen2.5-0.5B-Instruct --no-open
```

Use `--tiny` only to test transport with generated weights. The dashboard listens
only on the local machine and stores a sanitized local history. It does not emit
a canonical benchmark evidence record. A local run does not measure a multi-host
or wide-area deployment, model quality, energy use, price, adversarial behavior,
or production non-collusion.

Continue with [benchmarks](/sdk/research/benchmarks/) and [metrics](/research/records/metrics/) before comparing results.
