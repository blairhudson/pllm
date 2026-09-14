# Developer experience and documentation

Python exposes immutable configuration, discovery, experiment creation and public result objects. Rust performs model-data work, compilation, preparation, transport, inference, benchmarks and security instrumentation. Importing the DX package or reading a research card does not require a native extension; running a native benchmark does.

## Included, executable staging API

```python
from pllm_uplift import native_benchmark

result = native_benchmark(dim=64, lanes=18, repeats=7, bits=24, tile=32)
print(result["scalar_ns"])
print(result["tiled_ns"])
```

Build the extension first. This is a public synthetic modular-matrix benchmark, not a private model. JSON configs and cards can be inspected without native execution. The package's fixture/solver assurance command runs only its own explicit models, not whichever PLLM version happens to be installed.

## Intended production workflow to implement

`pllm model inspect`, `pllm methods list/show`, `pllm bench run/search/compare`, `pllm assure run`, `pllm research reproduce/cite` and `pllm serve` share the same model, plan, method and evidence registry. Exact command names may be reconciled with current APIs during repository inventory. They are not silently installed by this handoff.

An experiment resolves once into a public lock and separate role bundles. A user can clone public parameters for a search, but cannot clone one-time secrets. Results include eligibility failures instead of disappearing candidates. Report methods support JSON/Markdown/CSV and citations generated from the executed plan. No caller should need to infer the privacy boundary from an API name.

## sklearn-style method page

For a masked-linear method, explain a small public test example: modulo256, W=3, x=5, r=9, s=2. Preparation installs c=25; the client sends u=252; Inference returns v=13; the client adds s to recover15. Then explain who sees each value, why fresh independent masks matter, and what metadata remains visible. This example is arithmetic, not a security parameter recommendation. Cite Slalom and identify PLLM's changed trusted-client/dealer boundary separately.

Every page should then provide source/version, implemented construction/omissions, parameter definitions, numerical envelope, topology, conversions, preparation costs, privacy/proof status, known attacks, one executable fixture and reproduction artifacts. Do not copy a paper's reported speedup into a current performance badge.

Use the registry to generate CLI output, parameter docs, bibliography and provenance. Narrative explainers may be handwritten; measured tables come from immutable evidence files. Preserve the existing docs site framework rather than creating a migration dependency. Markdown/MDX and JSON remain usable by developers and agents offline.

## Errors and unavailable capabilities

Unknown models, kernels or protocol transitions fail with an actionable capability report. A missing native extension never invokes a NumPy implementation in a timed path. A missing proof review does not prohibit clearly labeled research, but blocks claims/defaults requiring that assurance. A paper available only as an abstract stays acquisition-gated; unsupported functionality is not filled in by a plausible invention.
