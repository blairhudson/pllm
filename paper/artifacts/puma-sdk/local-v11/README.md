# Guarded piecewise activation admission

This slice supplies bounded native guarded SiLU/GeLU programs and selects guarded
SiLU in public-weight PUMA decoders. Pinned Qwen source publication clears the
first MLP, then rejects layer-one RMSNorm. Full-checkpoint execution remains blocked.

## Reusable implementation

`Program::activation_guarded` evaluates the original shared comparisons before
the polynomial powers. One fresh exact product selects either the original input
or zero for those powers. Final shared selection still returns the original zero
tail, retained-domain polynomial, or unmodified identity tail. The coefficients,
endpoint inclusivity and Q18 stochastic equations remain unchanged. This adds no
runtime range-check acceptance or clipping of accepted activations.

Public integer partitions bound every polynomial's possible rounding outcomes.
They inspect no private values and add no runtime work. The changed instruction
digest charges the guard product; 128-lane and 256 KiB primitive limits remain.
Eager primitives and private-weight execution retain their previous identities.

Independent integer controls cover both activations, boundaries and neighbors,
wide signed tails, output bounds, PRSS/work cursors, cancellation, allocation
failure and erasure-before-refund. Generated Qwen2/Qwen3/Llama sources use wide
gates with a separate signed-head oracle and private feedback. Qwen2 additionally
traverses three authenticated SDK workers and checks session retirement.

## Pinned Qwen gate

The ordinary SDK source publisher uses the locked 4+2 Experiment from
`../local-v7/qwen-memory-preflight.json`, four fixed public zero IDs, fresh
whole-topology memory admission and the ordinary host-pressure watchdog.
Checkpoint values are read; no workers or online decoder are launched.

Admission now reaches selection zero, step 29, layer-one `RmsNorm`, with encoded
residual interval `[-343639410012,343639400796]`. Source publication observes zero
swap growth. Its CPU and process peak cover admission and its launcher, not
response costs or a client-memory tier.

Remaining qualification includes whole-model ranges, complete-program work,
independent hidden-state correctness, source-model quality and matched response
benchmarks. The separately recorded 16+2 lookup-only work rejection remains.
Protected runtime range checks would change acceptance and need a distinct
approved contract; this slice does not implement them.

## Evidence and replay

The final installed-wheel run passes **330 Python tests** with seven optional
skips. Native validation passes **159 checks**: 85 core PUMA tests, 26 compiler
tests, 43 allocation/multiply/PRSS/truncation integration tests and five doctests.
Clippy and benchmark coverage pass. All 367 installed files match the wheel before
and after validation; all 791 recorded inputs remain unchanged.

Wheel SHA-256:
`7575bf4dfad1706c03537deee97da22519a838adfa82139596d1ea47bd583884`.

`summary.json` derives counts from the original JUnit and native logs. `validation/`
retains installed-file comparisons, dependency versions, standard test commands
and the source-publication receipt. Its small runner replays the retained ordinary
SDK validation driver from `../local-v10/validation/run_validation.py`, with the
recorded hash and build-path substitution in `validation/base-harness.json`.

`build/` retains the guarded cached-local-build receipt, not a clean release
attestation. `inputs.tar.gz` preserves every tested input in
`validation-inputs.json`; the archive is checked against that manifest.
`package-delta.json` compares all installed files against the prior slice.

`old-wheel/tests.xml` retains three expected SiLU admission failures from the
earlier installed wheel. That diagnostic has no separate per-run installed-file
comparison; it adds no passing cases or performance evidence. Counts from this
slice overlap earlier validation and must not be added.

Verify the retained bytes from this directory:

```sh
shasum -a 256 -c SHA256SUMS
```

Rebuild/install the preserved inputs, then run the commands in
`validation/validation-commands.json`, adjusting interpreter and output paths.
Python uses the installed wheel with `-I`; the repository supplies test helpers.
Source publication requires the pinned checkpoint and fresh memory/disk admission.
Historical receipts authorize no new launch. Full equations and scope:
[guarded activation contract](../../../../docs/research/puma-truncation-reference.md#guarded-piecewise-activations).
