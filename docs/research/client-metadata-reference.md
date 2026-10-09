# Client construction attribution

The load-ready-artifact frontier first measures avoidable construction rather
than treating all client RSS as metadata. The installed
`BenchmarkTarget.response(experiment, client_metadata=True)` executes ordinary
source validation, compiler binding, preparation-demand sizing and runtime
construction with a fixed tracing owner.

Each phase records its new retained and peak Python-traced allocation, CPU and
wall time, plus recursively deduplicated public graph/binding object bytes.
The object walk excludes tensor storage. Native allocation, tokenizer/weight
owners and allocator fragmentation are separate. Construction traces include
temporaries allocated by the whole phase; their byte count is not an estimate
of removable metadata. Nested compile work is charged to the outer binding phase.

Tracing changes runtime costs and has a distinct measurement identity. These
responses cannot rank against untraced serving measurements. Hooks and tracing
ownership retire on success or failure; an existing tracer rejects admission.
Mutation of source-bound local tensors still fails ordinary revalidation.

## Validation and replay

The generated checkpoint passes the complete cold/repeat SDK target through
separate prepared roles. Native binding identities are preserved, invalid tensor
contents reject, and instrumentation restores on failure. Replay:

```sh
python -m benchmarks.research.client_stage_ownership --metadata --tiny \
  --processes 1 --workload short --output target/client-metadata-tiny.json
```

The real-checkpoint gate uses the same maintained prepared artifacts:

```sh
python -m benchmarks.research.client_stage_ownership --metadata --processes 3 \
  --workload medium --input-bound 384 --serial-issuance \
  --artifacts target/prepared-frontier-artifacts \
  --output target/client-metadata-qwen25.json
```

The separate untraced MLP screen counts 3,955,137 unique public metadata bytes per
runtime. It does not measure construction peaks or justify precompiled artifacts.
Source verification, artifact compilation/publication, parser/native ownership
and fresh-client whole-response RSS remain required before selecting a load-ready
artifact implementation. No runtime validation is bypassed by this target.
