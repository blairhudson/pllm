# Predictive lossless public weight coding

Status: **complete checkpoint codec screen; every predictor loses to zlib1**.
Twelve fresh serial workers verify 1,633,615,872 integer matrix entries and every
original row scale. All modes decode exactly, with zero observed swap growth.
Evidence: `docs/evidence/research-predictive-weight-codec.json`.

## Measured isolated result

Three processes per frozen mode cover the complete tied token/head matrix. The
control needs no new descriptor; each transformed candidate pays its descriptor,
indices and any dictionary. Unchanged scales remain additional in all modes.

| Mode | Charged matrix artifact, MB | Growth versus zlib1 | Median worker CPU, s | Median process peak RSS, MB |
| --- | ---: | ---: | ---: | ---: |
| zlib1 | 121.495 | — | 12.860 | 145.16 |
| row-delta | 127.210 | 4.70% | 13.273 | 147.05 |
| token-parent | 121.718 | 0.18% | 13.514 | 149.62 |
| global-token-dict512 | 129.311 | 6.43% | 21.488 | 158.38 |

Local token parents cover only 4.12% of rows. The global dictionary covers 99.68%,
but its residuals compress worse and its independently checked delivery costs
419,279 bytes plus descriptor metadata. Higher coverage does not imply prediction
quality. Keeping all other historical bodies fixed would grow the anchor pair
0.045% for token-parent, 1.16% for row-delta and 1.58% for the global dictionary.
Those are counterfactual projections, not new serving measurements.

Worker CPU/RSS includes source import, identity scans, quantization and independent
oracle work as well as encoding/decoding. No standalone thin-client decode or
whole-response saving follows. The tested fixed predictors fail their byte gate;
they do not reject every possible lossless learned or numerical predictor.

## Scope and pinned object

- Qwen/Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, W8A8.
- Complete tied `model.embed_tokens.weight`: 151936 × 896 signed int8 values,
  136134656 matrix bytes. One tied token/head object, not two distributions.
- Every row scale stays exact and unchanged: 607744 little-endian float32 bytes.
  BF16 source checkpoint remains unchanged and retained. No training, numeric
  weight changes, private text queries, or runtime/serving integration.
- Fixed 1024-row frames (last frame shorter), same shape and boundaries for
  every mode. Matrix payload per full fixture frame is 917504 bytes.

## Candidates and strongest control

`scripts/probe_predictive_weight_codec.py` freezes four independent modes:

1. **zlib1**: unchanged matrix bytes, zlib level 1, current runtime's `>II`
   raw-size/encoded-size framing. No transform, dictionary, or predictor.
2. **row-delta**: row zero predicts zero; each later row predicts previous
   original row. Exact byte subtraction modulo 256 before zlib1. No index bytes.
3. **token-parent**: public tokenizer strings choose longest earlier exact
   proper prefix or suffix token in the **same frame**. Maximum 64 characters
   searched on either side, maximum 256-character eligible tokens. Equal
   lengths prefer prefix, then earliest parent. Missing/padded/long tokens
   predict zero. Subtract parent original row modulo 256. Store one signed
   little-endian int32 index per row (`-1` means zero); indices and residuals
   are compressed together with zlib1. All index bytes are charged, including
    rows with no parent. No numerical fitting or hyperparameter search.
4. **global-token-dict512**: select up to 512 public token rows deterministically
   by proper prefix/suffix coverage across the complete public vocabulary.
   Candidates have 1–64 characters; each eligible target (at most 256 characters)
   probes at most 128 proper affixes. Count each parent at most once per target.
   Rank coverage descending, then original token id ascending. Selection uses
   no numeric weights, fitting, compression results, or tuning. For each target,
   predict with longest selected exact prefix/suffix (including whole-token
   match); equal lengths prefer lowest dictionary position. Use zero when no
   match. Distant ancestors are allowed: dictionary rows arrive independently
   before all matrix frames. Fixed coefficient one, no shrink fitting.

   Dictionary integers are byte-identical W8 rows, independently checked against
   raw BF16 + NumPy reference. Deliver them in `PWDIC001` layout: 20-byte `>8sIII`
   magic/row-count/column-count/frame-count header, followed by bounded zlib1
   frames. Descriptor charges selected original row ids, dictionary raw SHA-256,
   and layout identity. Matrix frames charge int32 dictionary indices plus exact
   modulo-256 residuals. No dictionary scales are needed for integer prediction;
   all original scales remain unchanged. Report actual parent coverage separately
   from coverage scores used in dictionary selection.

Frame-local mode remains an ablation, not a verdict on distant BPE parents.
Global candidate retains bounded decoder memory while including distant public
ancestors. Both are token-structure informed, not token-id row-delta or bitplane
repackaging. All choices are fixed before checkpoint cohort.

## Reversibility, bounds, and costs

Local parents must be `-1` or an earlier row in their frame. Global indices must
be `-1` or refer to a previously delivered dictionary row. Decode uses one
sequential pass, no recursion, no external tokenizer/checkpoint,
and no prompts. Cycles, forward indices, invalid dimensions, extra bytes,
truncation, corrupt streams, extra zlib members, and inflated-size mismatches
fail closed. External shape/mode descriptor is required; per-frame lengths
alone are not an authenticated artifact identity. Cryptographic object hashes
must be checked against a trusted descriptor before use. zlib checksum alone
does not authenticate maliciously rewritten values.

API bounds: at most 1024 rows, 4096 columns, and 1 MiB matrix bytes per frame;
Both indexed modes add at most 4096 inflated index bytes. Encoded body bounded by
declared inflated size plus 64 KiB. Inflate output capped at expected size + 1;
no unbounded `zlib.decompress`. Local decode buffers are O(frame bytes), below
8 MiB excluding interpreter/dependencies and caller's retained input/output.
Global persistent dictionary is at most 512 × 4096 = 2 MiB (458752 bytes for
fixture), decoded from at most two independently bounded frames. Global matrix
decode retains only dictionary + one frame, never complete matrix or tokenizer.
Conservative codec scratch envelope is 16 MiB, including dictionary inflation
copies; dictionary digest is verified before any matrix frame uses it.
Local setup retains at most 1024 token entries. Global setup is bounded to
200000 public tokens, at most 128 affix probes per target, at most 512 selected
rows, and one deterministic ranking. Tokenizer JSON is limited to 32 MiB; decoded
Python JSON/string overhead is separately present in fresh-process peak RSS.

Local modes distribute no standalone dictionary. Global mode pays full framed
dictionary + metadata + indices **once per cold/exact-repeat pair**. Public
tokenizer is encoder-side retained
source storage, not silently required decoder state. Tokenizer parsing CPU,
predictor construction CPU, retained source bytes, raw index bytes, compressed
frame bytes, canonical descriptor bytes, encode/decode CPU, source/reference CPU,
identity-hashing CPU, pre-worker import CPU, total process CPU, parent-observed
fresh-process wall time, and peak RSS are separate fields. No setup amortization.
Dictionary selection, source reads/quantization/independent checks, dictionary
encode/decode CPU, integer payload bytes, header bytes, framed distribution bytes,
and canonical metadata are explicit fields. Source read counts are logical
successful BF16 bytes for runtime and reference inputs, not measured OS I/O;
failed adapter reads and filesystem readahead are not invented. No new predictor
distribution at repeat time is assumed.

Canonical descriptor includes schema, mode, full shape, row-frame size, weight
and scale SHA-256, selected dictionary ids and dictionary commitment. Candidate
artifact storage includes dictionary layout plus descriptor plus all
framed bytes; unchanged scale storage is reported separately. Files are not
persisted by workers, so storage counts are arithmetic, not storage throughput
measurements. Peak RSS includes imports, tokenizer, source mmap, quantization,
independent reference, and encode/decode overlap. This is **not** a decoder-only
RSS measurement. Decode timing follows encode inside a fresh worker, so does
not establish cold standalone decoder latency.

## Independent complete-object validation and identity freeze

Each worker streams every matrix row, uses actual imported runtime quantizer,
and compares every integer and scale byte with a separately implemented NumPy
max-absolute/127, round-to-nearest-even reference. Independent bounded raw
safetensors access validates a ≤1 MiB JSON header, BF16 dtype, rank-two shape,
object offsets and file bounds, then reads exact contiguous little-endian BF16
rows. Conversion uses uint16→uint32 bit shift/view, not NumPy BF16 support or
Torch. Installed `SafeTensorStore` is attempted for runtime input; BF16 adapter
rejection alone triggers bounded raw-row fallback, recorded in worker output.
Reference input is always independently raw-read. Complete original BF16 tensor
bytes are hashed without float32 reinterpretation. Dictionary rows and scales
are independently verified, with separate source/dictionary/scale digests.
Complete raw, decoded, independent-reference, scale, BF16-object, and framed
artifact hashes are recorded. Native/reference disagreement blocks completion;
never relabel a differing matrix as successful lossless compression.

Resolved original checkpoint/tokenizer/config files, original control evidence,
authoritative `research-all-link-frontier.json`,
probe, framing/control helper, Python executable, actual native extension, zlib
extension (or its owning interpreter when built in), imported runtime paths, and relevant dependency package code/native
trees are hashed before and after each worker and the cohort. Installed package
versions, backend, Python and zlib identities are frozen too. Source names do
not point at generated temporary snapshots. Package-tree freeze covers lazy
runtime/dependency imports. Installed distribution is **pllm.run**, with no
source-tree metadata fallback. `PYTHONPATH` must be unset; replay imports installed
runtime/native, not `python/pllm`. Original checkpoint/config hashes must also
match authoritative frontier model identity. Shared runner checks BuildRef and host
pressure/swap growth under `~/.cache/pllm/frontier-probe.lock`.

## Accounting boundaries

Authoritative reconciled ledger: **493575247** bytes per cold/exact-repeat pair;
**124905091** public bundle; **90443518** inputs. Raw confirmation SHA-256 is
`9b54ee455e7307429d4cc1bf19c93f1c58ff3ef966c258ddebde2ec8c68c42db`.
Runner requires that hash, calls `conserved_control()`, compares its full result
with authoritative frontier `control`, and freezes both evidence files. All
projection arithmetic derives directly from validated control fields; no
repeated ledger literals in code. Candidate matrix cost (including dictionary
and metadata when applicable) is compared against same-frame plain
zlib1. Existing zlib1 control needs **no new descriptor**, whereas transformed
candidates pay their full descriptor: stronger than symmetric descriptor
charging. Matrix-only savings translate arithmetically to a whole-bundle and
pair counterfactual with all other bytes unchanged, including unchanged scales.
Negative savings are retained. These are **not measured serving responses**.
Real bundle object/frame alignment differs from isolated row frames and needs
an integrated benchmark before accepting that projection. No assumption of
10–20% bundle compression is used in scoring.

## Functional proof and serial handoff

Tiny tests (no checkpoint/network):

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath=. tests/test_predictive_weight_codec.py -q
```

44 passed in 0.30 s on final tiny functional run. Tests
cover all modes, random bytes, modulo wrap, token prefix/suffix selection,
maximum row count, malformed indices, bounds, truncation, corruption, trailing
streams, inflate bomb, independent reference scale bytes, global coverage and
distant parents, dictionary header/digest/budget failures, two-frame maximum
dictionary, control/hash binding, projection losses, fixed 12-worker schedule,
and generated 3×2 BF16 safetensors rows. Installed NumPy safetensors adapter
accepted this fixture; raw fallback is also forced and verified synthetically.
Installed native quantizer matches independent fixture integer/scale bytes.
Largest payload is 2 MiB; tiny functional work stays well below 512 MiB.

Replay the checkpoint/performance cohort serially:

```sh
target/frontier-sdk-env/bin/python scripts/probe_predictive_weight_codec.py --processes 3 --output target/predictive-weight-codec.json
```

Run with `PYTHONPATH` unset. Exactly twelve fresh workers: four frozen modes ×
three replicates, rotated mode order.
`fresh_workers()` admits 1 GiB post-reserve headroom and applies shared lock and
host/swap watchdog. No parallel workers, downloads, installs, builds, or fit
search. Local-only snapshot lookup; missing cached tokenizer/checkpoint or
missing native extension is a blocker, not permission to download/build.
Per-worker timeout 600 s; worst worker deadline envelope 120 min, not a runtime
estimate. The historical pre-run 0.5–1 GiB planning envelope is not measured RSS;
actual process medians appear above. Identity hashing rereads complete checkpoint each time and
can dominate. RSS pressure includes file-backed source pages, not just codec
scratch. Global selection retains public token strings, coverage array and
bounded ranking state, not numeric full matrix. Fixture dictionary adds <0.5 MiB
persistent integers; general codec scratch cap is 16 MiB. 1 GiB admission remains
conservative for tokenizer/import/reference/file-backed overlap, not a hard RSS
limit or measured guarantee. Host admission may reject cohort.

Use a fresh `target` output while probe source is frozen. Preserve completed
evidence and do not modify probe/helper/dependencies during the cohort. Cached
snapshot resolution requests required model/config/tokenizer files only; unrelated
uncached repository files are not a reason to download additional artifacts.
