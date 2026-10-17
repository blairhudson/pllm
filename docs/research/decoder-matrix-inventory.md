# Decoder matrix inventory

Status: **native metadata and installed Python binding checks pass**.
This inventory exposes the compiler's existing matrix groups without selecting
a numerical profile or a runtime composition.

## Contract

`pllm_compiler::lower_decoder_matrix_inventory` accepts a typed `DecoderPlan`.
`decoder_matrix_inventory_bytes` accepts bounded decoder-plan JSON. Both return
an immutable `DecoderMatrixInventory` with canonical bytes and a domain-separated
SHA-256 digest. The schema and digest domain are `pllm.decoder_matrix_inventory.v1`.
The internal PyO3 binding is declared as
`_native.decoder_matrix_inventory(plan_bytes) -> (canonical_bytes, digest)`.
It releases the interpreter lock during native extraction.

Both phases contain ordered groups with their semantic operator, layer, input,
orientation, logical shape, source members, bias shapes, and output slices.
The existing scheduler's grouping function supplies membership and order.
No adapter-specific operation names or source paths determine grouping.

Matrices use `[output, input]`; token tables use `[vocabulary, feature]`.
Each member retains its source artifact and its logical output offset.
Ownership is keyed by artifact kind and source ID. A tied source therefore does
not merge a float32 lookup table with an i16 head representation.

The inventory checks declared references, dimensions, biases, phase agreement,
and an empty transformation list. It accepts batch one, operation-produced
linear/head inputs, and a nonempty contiguous layer sequence starting at zero.
State-only linear inputs and unsupported matrix attributes reject.
An empty transformation list is a declaration, not proof of an unmodified source.

Admission precedes grouping and output serialization:

- Input and canonical output: at most 8 MiB each.
- Operations: 4,096 per phase; states: 1,024 per direction and phase.
- Inputs: 16 per operation; rank: at most five; object fields: 64.
- Strings: 1,024 UTF-8 bytes each and 1 MiB total, including keys.
- Whole-document JSON values: 131,072; depth: 24 with the root at zero.
- Dimensions and concatenated widths: `2^20`; individual shape products:
  `2^40`; summed source-use elements: `2^48`.

Typed and JSON admission count the same schema fields, null options, enum
strings, keys, and value nodes. JSON preflight also rejects duplicate keys before
constructing a plan. A conservative output bound precedes allocation of groups.
No tensor contents are read or allocated.

## Evidence and limits

Fourteen inventory tests and 22 scheduler tests passed. These include exact
depth/node/string boundaries, malformed references and geometry, renamed
artifacts, tied owners, and source-order grouping. Three captured legacy schedule
digests remain unchanged. The pinned Qwen configuration yields 96 body groups,
one full head, and one lookup table from configuration alone.

The checkpoint scanner now consumes this native metadata and independently checks
plan/configuration digests, phase fields, complete operation coverage, source
headers, biases, orientations, concatenated dimensions, and output slices.
It reports conditional grouped publication geometry separately from its earlier
conditional unfused geometry. The 45 Python
scanner/metadata tests initially passed against the existing installed package
with binding doubles. A further nine malformed-geometry controls brought that
suite to 54 passing tests; three generated-family native checks were deliberately
excluded. Those tests do not establish execution of the new binding.

The two binding tests and three generated-family checks now pass in the sixth
cached wheel's isolated environment, within a 67-test suite. Earlier fixture
failures and their corrected Qwen3 configuration are preserved. The
[accepted archive](../../paper/artifacts/native-inventory/2026-10-17-installed-v6/README.md)
binds the build, wheel hash, tests and completed source scan. This is a cached
local build, not a clean-release attestation.

Historically, the third cached wheel attempt stopped on 704,512 bytes of reported
`sout` growth, with clean retirement and no source change. It produced no wheel.
The earlier headroom rejection and cleanup-permission failure remain separate
attempts. On macOS that psutil counter measures pageouts, not compressor swapouts;
these increments cannot establish actual swap activity or attribute it to compilation.
Original receipts and available driver snapshots are retained in the
[blocked-build archive](../../paper/artifacts/native-inventory/2026-10-17-blocked-wheels/README.md).
A fourth attempt followed a separate 30-second quiet preflight with at least
12,448,366,592 available host bytes. Its unchanged guard then stopped on 65,536
bytes of reported `sout` growth. Cleanup passed and source snapshots matched, but
no wheel was produced. The [follow-up archive](../../paper/artifacts/native-inventory/2026-10-17-blocked-followup/README.md)
retains both observations. A quiet preflight does not establish build admission
through completion.

This metadata does not authenticate checkpoint contents, validate local operator
semantics, admit a source numerical graph, prove state handoff, or select GKM
execution. The plan's configuration digest remains a caller declaration.
The separate third [checkpoint scan](gkm-checkpoint-bounds.md) passes source-only
numeric admission. No matrix publication or GKM decoder execution was performed.
