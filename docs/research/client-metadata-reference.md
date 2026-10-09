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

## Three-client checkpoint attribution

Three fresh installed clients each complete a 150+8-token cold/repeat Qwen2.5
pair under native shared 100/40 caps. All six requests pass, with matching output
commitments and zero observed swap growth. The instrumented build is
`0201f1d6947159b1c5c528ac9648a8154258f24e2cbba13d1344494f78138c0f`.

The following medians summarize individual observed phases, not simultaneous
owners. MB are decimal. Each client performs eight compiler-binding phases,
four preparation-demand phases and two runtime-binding phases.

| Phase | Observations | New traced peak MB | New traced retained MB | Unique public metadata MB |
| --- | ---: | ---: | ---: | ---: |
| Compiler binding | 24 | 9.448 | 2.424 | 2.485 |
| Preparation demand | 12 | 2.757 | 0.059 | 0.00974 |
| Runtime binding | 6 | 9.471 | 4.238 | 3.962 |

The largest observed construction increment is **9.580 MB**; a complete runtime's
unique public metadata is **3,961,713 bytes**. Summing phase peaks or repeated
metadata counts would invent an owner that was not measured. Traced phase CPU
is about 3.5 seconds per binding and cannot predict untraced validation savings.

**Decision:** metadata-only replacement is not a credible tenfold client-RAM
explanation for this workload. The screen does not justify a new load-ready
artifact format. Repeated binding is a separately measurable CPU opportunity,
requiring an untraced matched ablation before any benefit claim. The target proved
useful precisely because it distinguishes retained metadata, construction overlap
and instrumentation overhead.

Evidence: `docs/evidence/research-client-metadata-qwen25-cohort.json`, with its
`.plan.json` and `.frontier.json` companions. Replay uses the command above;
historical untraced MLP measurements retain their original installed build.
