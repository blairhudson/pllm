# Current support

What you can use today, what remains experimental, and what is not yet supported.

[View canonical HTML](https://pllm.run/sdk/reference/status/)

Document ID: `pllm.docs.reference.status`  
Release: `0.1.0`  
Build: `sha256:65f4ad316621284cc28b60b1a825a6d9c6d1f108cbb5032aee0983de1e5c4966`  
Source hash: `sha256:5fe45104a841c46bde341a57e65c97700467a75183f0fa1447de7d1db54cffc5`

Checked 14 September 2026.

The CLI can inspect and export public configuration, list built-in components,
inspect research records, and run a headless loopback benchmark.

The Python package exposes semantic model planning and narrow native benchmark
and assurance APIs. These interfaces do not yet provide complete private model
execution from a `ModelPlan`.

The CLI does not currently manage remote services, private chat, preparation, or
canonical evidence-producing benchmark runs.

## Status labels

PLLM uses six labels throughout the documentation:

| Label | Meaning |
| --- | --- |
| Available | Implemented and usable within the stated limits |
| Experimental | Implemented, but not ready for production use |
| Planned | Intended work with no available implementation |
| Not supported | Outside the current implementation |
| Not evaluated | Implemented or proposed, but not measured for this claim |
| Not applicable | The claim does not apply to this item |

## CLI

| Command | Status | Current limit |
| --- | --- | --- |
| `config show`, `config export` | Available | Validates public `Experiment` data; Python targets require explicit trust |
| `components list`, `components show` | Available | Reads built-in component metadata only |
| `research sources/methods/recipes list/show` | Available | Reads records without running research workflows |
| `dev dashboard` | Experimental | Runs all roles on one machine; does not create benchmark evidence records |
| `benchmark run` | Experimental | Runs all roles on one machine and writes a sanitized diagnostic report, not a canonical `EvidenceReport` |
| `init`, `--set` | Not supported | Profile catalog and typed overrides are incomplete |
| `model lower` | Not supported | The CLI output contract is incomplete |
| `plan check`, `plan compile`, `plan show` | Not supported | Plan locking, artifacts, and compatibility checks are incomplete |
| `prepare`, `run`, `chat` | Not supported | The CLI does not manage prepared material |
| `serve`, `party serve` | Not supported | Role authentication and orchestration are incomplete |
| `benchmark search`, `benchmark compare` | Not supported | The CLI does not query or compare canonical evidence records |
| `assure run` | Not supported | The CLI assurance interface is incomplete |

The [CLI reference](/cli/reference/) contains the exact available commands.

## Python support

| Capability | Current status |
| --- | --- |
| Immutable `Experiment` configuration | Available for the documented schema |
| Semantic model adapters | Available for the listed Qwen2, Qwen3, Qwen3.5, Phi-4-mini, and Gemma 4 text configurations |
| MPCache plan transformation | Experimental for compatible Qwen2 plans; Gemma plans are rejected |
| Compiler coverage report | Available; the current named profile remains incomplete |
| Native compile, benchmark, and assurance APIs | Available for their documented narrow scopes |
| Complete private generation from `ModelPlan` | Not supported; Qwen3.5 recurrent operators are missing from the compiler |
| Generation quality for semantic adapters | Not evaluated |
| Privacy evidence for generated component descriptors | Not evaluated |
| Generic remote role deployment | Not supported |
| Production deployment maturity | Not evaluated |

The package still exposes older runtime and client interfaces. Their presence does
not mean that complete plan-based deployment is supported. See the generated
[Python API inventory](/sdk/reference/python/pllm/).

## Python SDK example

```python
from pllm.components import get_component

component = get_component("pllm/cpu")
assert component.lifecycle_phase == "compilation"
```

Static discovery reports availability, not production maturity or complete private execution.

API: [`pllm.components.get_component`](/sdk/reference/python/pllm/#objects-and-signatures)
