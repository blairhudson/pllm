# Guarded shared softmax admission

This slice adds bounded secret tail selection to public-weight PUMA softmax.
Pinned Qwen source publication now clears its first softmax and rejects the
following RMSNorm. Complete-checkpoint execution remains unqualified.

## Reusable implementation

`Program::softmax_guarded` uses the original shared row maximum and strict
`z > -14*2^18` predicate. One additional exact, untruncated shared product
replaces clipped inputs with zero before the existing exponential operations.
The final predicate multiplication still gives exact zero tails. Retained lanes
keep the original constants, five-bit rescale, five sequential square/truncation
steps, denominator, reciprocal and output rescale.

The native bound follows from the predicate equation and the public input
interval. Private values never select a public range. Program identity binds the
changed schedule; sequence/AES pricing charges the extra product, and every
invocation consumes fresh party-local material. The eager primitive and default
private-weight compiler keep their previous identities and rejection behavior.

Independent integer recipes replay each schedule's own truncation tape. Checks
cover the strict-cutoff neighbors, non-power-of-two and constant rows, large
public intervals, 128/512-value rows, all-event cancellation, and allocation-index
failure with erasure-before-refund. Primitive work and the original 256 KiB party
payload ceiling remain enforced.

Generated Qwen2/Qwen3/Llama regressions use wide nonzero Q/K and exercise private
feedback. Their V/O/MLP weights remain zero, preserving an independent signed-head
oracle; primitive tests separately check every softmax output. The Qwen2 control
also runs through three authenticated SDK workers. All three new regressions
reject softmax on the previous installed wheel; `old-wheel/` retains that negative
control, including the installed package hashes. It supplies no passing count.

The full equations and ownership contract are in the
[reference](../../../../docs/research/puma-truncation-reference.md#guarded-shared-softmax-tails).

## Pinned Qwen source publication

The ordinary SDK publisher uses the same locked four-input/two-output Experiment
as `../local-v7/qwen-memory-preflight.json`, four fixed public zero IDs, fresh
whole-topology memory admission and the ordinary memory watchdog. It decodes
source values but launches no role workers or online decoder.

The first rejecting operator is now layer-zero post-attention `RmsNorm`, semantic
step 22 in selection zero. Its public encoded input interval is
`[-1571860067, 1571850851]`. `validation/qwen-program-admission.json` retains the
original rejection and zero observed swap growth. CPU and process peak cover
source admission and its validation launcher, not serving or a client-memory tier.

Full-model range proofs, complete-program work, hidden-state oracles, source-model
quality and matched response costs remain separate gates. This slice establishes
no full-checkpoint request or performance improvement.

## Evidence and replay

| Check | Result |
| --- | --- |
| Installed-wheel Python cases | 321 passed; seven optional skips |
| Native PUMA library | 80 passed |
| Native replicated compiler | 24 passed |
| Allocation, multiply, PRSS and truncation integration | 43 passed |
| Ownership doctests | Five passed |
| Clippy and public coverage inventory | Passed |
| Installed package files | All 367 match before and after validation |
| Source/test inputs | All 791 match before and after validation |

The 152 native checks include the five doctests. Clippy retains its recorded
`needless_range_loop` and `nonminimal_bool` allowances. The seven optional skips
need Metal or explicitly configured checkpoint paths; the pinned source-publication
diagnostic has its separate actual-checkpoint receipt. Only the native extension
changes among installed package files.

`summary.json` derives counts from the original JUnit and native logs.
`validation/` retains commands, package-file comparisons, dependency versions,
source-publication receipt and the original validation launcher. `build/` retains
the guarded cached-local wheel receipt. The source/test manifest and
`inputs.tar.gz` preserve the tested input bytes independently of the build cache.
`package-delta.json` compares installed package contents with the previous slice.

Verify these read-only artifacts from this directory:

```sh
shasum -a 256 -c SHA256SUMS
```

Rebuild/install the preserved source inputs, then run the commands in
`validation/validation-commands.json`, adjusting interpreter and output paths.
Python checks use the installed wheel with `-I`; the repository supplies test
helpers. The archived launcher uses ordinary SDK source publication and requires
the pinned checkpoint plus fresh host/disk admission. Historical receipts do not
authorize a new launch. Counts overlap earlier slices and are not additive.
