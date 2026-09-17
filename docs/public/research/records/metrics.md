# Metrics

Choose clear performance, resource, quality, and privacy measurements for a PLLM experiment.

[View canonical HTML](https://pllm.run/research/records/metrics/)

Document ID: `pllm.docs.metrics`  
Release: `0.1.0`  
Build: `sha256:426652b6512bb11e794ef7caf6e150d2b19f0c9f7b933d041295b3bacebb441a`  
Source hash: `sha256:193a7b72d5185ed5e1d53cad4f962bbd540874b060b24ac3b628cc01dd041fba`

No single number describes system performance. Report tokens per second, time to
first token, per-token latency, peak and retained memory, CPU time and use,
online and offline network traffic, disk use, preparation work, and failures.
Energy and price require measurements from the stated environment.

Define each metric's scope and denominator. Keep prefill separate from decode,
client from provider, offline from online, cold from warm, and useful output from
attempted work. Record model quality and privacy assurance separately from
performance.

Telemetry must never include prompts, generated text, token IDs, activations,
seeds, masks, or credentials.
