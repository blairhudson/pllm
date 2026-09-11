# Validation

This file records checks executed against the current source checkout on
2026-09-11. It is evidence for this revision, not a universal performance or
security result.

## Executed checks

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` | **400 passed** |
| `PLLM_FORCE_SCALAR=1 .venv/bin/python -m pytest -q` | **400 passed** |
| `.venv/bin/python -m pytest -q tests/test_dashboard.py tests/test_dashboard_history.py` | **16 passed** |
| `cargo test --workspace` | **11 passed** across four Rust suites |
| `cd docs && npm run check:content` | **30 MDX pages**, 2 standalone pages, **0 errors** |
| `cd docs && npm test` | **18 passed**, including navigation and internal-link checks |
| `cd docs && npm run typecheck && npm run build` | **Passed**, 35 static routes |
| `scripts/build_paper.py` | **7 pages**; extracted arXiv archive compiled |
| `scripts/build_whitepaper.py --publish` | **2 pages** |
| `uv build && scripts/check_distributions.py dist` | Wheel and sdist passed |
| Isolated Python 3.11 wheel dashboard smoke | Completed schema-3 run; all roles sampled; clean single-signal exit |
| `docker build -f deploy/Dockerfile.site -t pllm-docs .` | Passed; 18 tests, typecheck, and 35-route static export ran inside image build |
| `pllm-docs` nginx smoke | Home, docs, research, whitepaper, PDF, and TeX routes returned `200`; zero nginx errors |

The focused 16-test run makes the dashboard/history evidence easy to reproduce.
The full native and forced-scalar runs cover the protocol, package, transport,
backend, and optional compatibility tests present in this checkout.

## What this supports

- Public-weight inference uses offline seeded preparation before `READY`.
- Corrections are pushed preparation-to-inference and acknowledged before seal.
- Online public inference is client-to-inference only; tested preparation-attempt
  counters remain unchanged during response execution.
- Row reservations, one-use tickets, burn-on-close, expiry, idempotent correction
  upload, compact prefill, and persistent decode are exercised.
- Public boundary matrices stay client-side and do not enter tested online request
  envelopes.
- `u16`, `u24`, and `u32` ring behavior and the Rust modular core have automated
  coverage.
- Dashboard tests cover role command defaults, authenticated telemetry ingestion,
  loopback Host/Origin binding, sanitized immutable SQLite history, pagination,
  cold/warm cohorts, monotonic timing, failed streams, and bounded shutdown.
- Documentation content, required journeys, links, and TypeScript compile cleanly.

## What this does not establish

- No real-checkpoint benchmark was executed for this overhaul.
- Tiny or mocked tests do not establish model quality or model-scale performance.
- Loopback tests do not establish wide-area latency, proxy behavior, operator
  isolation, secure erasure, host hardening, or production readiness.
- Tests do not prove non-collusion or malicious security. The current protocol
  assumes honest-but-curious, non-colluding preparation and inference roles that
  follow the lifecycle.
- The source tree has OpenAI SDK integration tests, but not an executed Agents SDK
  integration test. Test the pinned Agents SDK used by an application.
- Historical paper values, synthetic matrix measurements, and projected model
  throughput are not current end-to-end benchmark results.
- The site container ran only as a local loopback smoke; no registry push,
  external proxy, CDN, or production Pages deployment was tested.

## Reproduce

```bash
uv sync --locked --extra he --extra sdk
uv run pllm build
uv run pytest -q
cargo test --locked

cd docs
npm ci
npm run check:content
npm test
./node_modules/.bin/tsc --noEmit
```

Run the current real-role benchmark separately and record hardware, OS, source
revision, exact model revision, cache/inventory state, history schema, prompt and
output lengths, and command flags:

```bash
uv run pllm benchmark dashboard
```

Use `--tiny --history-db :memory:` only as an ephemeral transport/dashboard smoke
test. Random tiny output and timing are not real-model evidence.
