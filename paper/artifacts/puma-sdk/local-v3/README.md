# Compact PUMA compiler conformance

This slice changes public compiler ownership within the existing bounded
`PumaThreePartyCpu` composition:

- Checked strided descriptors replace expanded source-weight references and
  per-tile index lists. Only the admitted primitive tile is copied at runtime.
- Identical immutable tensor programs share one allocation. Each invocation
  still binds its own ordinal, fresh PRSS work and consuming protocol owner.
- Public construction views retire after their last semantic consumer.

Source shares, full-prefix recomputation, operation limits, primitive ceilings
and full-model source capacity retain their existing contracts. These checks
measure neither complete-client RSS nor a serving-speed improvement.

## Evidence

[`summary.json`](summary.json) records the installed-wheel, native and admission
checks. The original build and validation records are under `build-v1/`.

Native regressions cover six frozen Qwen2/Qwen3/Llama execution identities from
the previous installed quality wheel, shared-program ownership with complete
per-invocation work accounting, checked descriptor overflow and owner rejection,
nonzero private prefix dependence, and three-party matrix execution with one
party retaining the old expanded-gather representation. Rectangular, transposed,
partial-tile and long-contraction cases compare complete integer outputs.

The installed suite exercises ordinary three-worker SDK requests, private
feedback, gateway, CLI response/quality benchmarks and failure cleanup. Every
installed package file is compared with the wheel before and after validation.
Source and test inputs are likewise hashed at both boundaries. Suite counts
overlap previous conformance slices and are not additive.

The current ordinary quality command for pinned Qwen2.5 still exits with code 6:
`decoder source expansion budget exceeded or invalid`. Its public declaration
uses four input tokens and two allowed outputs; source admission precedes tensor
values and worker launch. The retained command, input hashes and structured
rejection establish a blocked cell, not a full-checkpoint quality result.

The guarded build reuses local Cargo state and dependencies. Its swap counters
cover the sampled build process tree only; tests and serving have separate scope.
This is local conformance evidence, not revision-bound release CI or independent
cryptographic review.

## Replay

From this directory, verify the retained records:

```sh
shasum -a 256 -c SHA256SUMS
```

`inputs.tar.gz` and `validation-inputs.json` retain the tested source and test
bytes. The local-v2 archive retains the preceding golden-reference build.
From the reconstructed repository root, build/install its wheel and run:

```sh
python -I -m pytest -q -o pythonpath=. -o junit_family=xunit1 \
  tests/test_replicated_*.py tests/test_reference_agreement.py
cargo test --locked --offline -p pllm-compiler --lib replicated_decoder
```

Exact arguments, dependency versions, wheel identity, file checks and lint
allowances remain in `build-v1/validation-commands.json` and the build receipts.
Full-Qwen execution still requires bounded checkpoint ownership, complete-model
range admission and the separate source-model quality gate.
