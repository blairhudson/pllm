# Publish the documentation

Validate and export a static site with no inference endpoint.


## Local content checks

From `docs/`:

```bash
npm ci
npm run check:content
npm test
```

These checks do not regenerate paper or historical evidence. A full framework
validation additionally runs:

```bash
npm run typecheck
npm run build
```

The static export is written to `out/`. It contains documentation, browser search,
and checked public assets only. It does not contain or proxy an inference service.

## Repository-path export

Test a project Pages path without hardcoding an owner:

```bash
NEXT_PUBLIC_BASE_PATH=/pllm npm run build
```

The configured base path prefixes framework routes, search, downloads, and links
inside repository-owned HTML. The Pages workflow obtains the actual path from
GitHub's Pages configuration.

## Deployment boundary

Configure **Settings -> Pages -> GitHub Actions** in the repository you control.
The supplied workflow builds and uploads a static artifact through the
`github-pages` environment. A successful local build is not evidence that a
remote deployment occurred.

Search executes entirely in the browser. Never put credentials, prompts, private
model weights, activation telemetry, root seeds, masks, or prepared inventory
under `public/` or `out/`. Historical paper/evidence assets have their own
canonical build and should not be regenerated during operational docs edits.
